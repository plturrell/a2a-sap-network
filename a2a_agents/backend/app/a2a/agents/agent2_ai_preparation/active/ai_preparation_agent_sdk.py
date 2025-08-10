"""
AI Data Readiness & Vectorization Agent - SDK Version
Agent 2: Enhanced with A2A SDK for simplified development and maintenance
"""

import asyncio
import uuid
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import logging

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


class BusinessContext(BaseModel):
    primary_function: str
    stakeholder_groups: List[str]
    business_criticality: float
    operational_context: str
    strategic_importance: float


class RegulatoryContext(BaseModel):
    framework: str
    compliance_requirements: List[str]
    regulatory_complexity: float


class SemanticEnrichment(BaseModel):
    semantic_description: str
    business_context: BusinessContext
    domain_terminology: List[str]
    regulatory_context: RegulatoryContext
    synonyms_and_aliases: List[str]
    contextual_metadata: Dict[str, Any]


class EntityRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    relationship_strength: float
    confidence_score: float


class VectorRepresentation(BaseModel):
    entity_id: str
    vector_embedding: List[float]
    embedding_model: str
    embedding_dimension: int
    semantic_tags: List[str]
    creation_timestamp: datetime


class AIReadyEntity(BaseModel):
    entity_id: str
    original_entity: Dict[str, Any]
    semantic_enrichment: SemanticEnrichment
    vector_representation: VectorRepresentation
    entity_relationships: List[EntityRelationship]
    ai_readiness_score: float
    quality_metrics: Dict[str, float]
    processing_metadata: Dict[str, Any]


class AIPreparationAgentSDK(A2AAgentBase):
    """
    Agent 2: AI Data Readiness & Vectorization Agent
    SDK Version - Transforms standardized financial entities into AI-ready semantic objects
    """
    
    def __init__(self, base_url: str, vector_service_url: str):
        super().__init__(
            agent_id="ai_preparation_agent_2",
            name="AI Data Readiness & Vectorization Agent",
            description="A2A v0.2.9 compliant agent for AI data preparation and vectorization",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.vector_service_url = vector_service_url
        self.ai_ready_entities = {}
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(4)  # 4 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            "vectorization_successes": 0,
            "semantic_enrichments": 0,
            "relationship_mappings": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8003'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing AI Preparation Agent resources...")
        
        # Initialize AI-ready entity storage
        storage_path = os.getenv("AI_PREPARATION_AGENT_STORAGE_PATH", "/tmp/ai_preparation_agent_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Load existing state if available
        await self._load_agent_state()
        
        logger.info("AI Preparation Agent initialization complete")
    
    @a2a_handler("ai_data_preparation")
    async def handle_ai_data_preparation(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for AI data preparation requests"""
        start_time = time.time()
        
        try:
            # Extract entity data from message
            entity_data = self._extract_entity_data(message)
            if not entity_data:
                return create_error_response("No valid entity data found in message")
            
            # Process entity for AI readiness
            ai_ready_entity = await self.prepare_entity_for_ai(
                entity_data=entity_data,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='ai_preparation').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='ai_preparation').observe(time.time() - start_time)
            
            return create_success_response(ai_ready_entity)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='ai_preparation').inc()
            logger.error(f"AI data preparation failed: {e}")
            return create_error_response(f"AI preparation failed: {str(e)}")
    
    @a2a_skill("semantic_enrichment")
    async def semantic_enrichment_skill(self, entity_data: Dict[str, Any]) -> SemanticEnrichment:
        """Enrich entity with semantic metadata"""
        
        # Extract business context
        business_context = BusinessContext(
            primary_function=entity_data.get('primary_function', 'Financial Data Processing'),
            stakeholder_groups=entity_data.get('stakeholders', ['Finance', 'Analytics', 'Compliance']),
            business_criticality=self._calculate_business_criticality(entity_data),
            operational_context=entity_data.get('operational_context', 'Core Banking Operations'),
            strategic_importance=self._calculate_strategic_importance(entity_data)
        )
        
        # Extract regulatory context
        regulatory_context = RegulatoryContext(
            framework=entity_data.get('regulatory_framework', 'Basel III'),
            compliance_requirements=entity_data.get('compliance_reqs', ['GDPR', 'SOX', 'Basel III']),
            regulatory_complexity=self._calculate_regulatory_complexity(entity_data)
        )
        
        # Generate semantic enrichment
        semantic_enrichment = SemanticEnrichment(
            semantic_description=self._generate_semantic_description(entity_data),
            business_context=business_context,
            domain_terminology=self._extract_domain_terminology(entity_data),
            regulatory_context=regulatory_context,
            synonyms_and_aliases=self._generate_synonyms(entity_data),
            contextual_metadata=self._extract_contextual_metadata(entity_data)
        )
        
        self.processing_stats["semantic_enrichments"] += 1
        return semantic_enrichment
    
    @a2a_skill("vectorization")
    async def vectorization_skill(self, entity_data: Dict[str, Any], semantic_data: SemanticEnrichment) -> VectorRepresentation:
        """Generate vector embeddings for the entity"""
        
        # Combine entity data with semantic enrichment for vectorization
        vectorization_text = self._prepare_vectorization_text(entity_data, semantic_data)
        
        # Generate vector embedding (placeholder - would use actual ML model)
        vector_embedding = await self._generate_vector_embedding(vectorization_text)
        
        # Create vector representation
        vector_representation = VectorRepresentation(
            entity_id=entity_data.get('entity_id', str(uuid.uuid4())),
            vector_embedding=vector_embedding,
            embedding_model="financial-bert-v1.0",
            embedding_dimension=len(vector_embedding),
            semantic_tags=self._generate_semantic_tags(entity_data, semantic_data),
            creation_timestamp=datetime.utcnow()
        )
        
        self.processing_stats["vectorization_successes"] += 1
        return vector_representation
    
    @a2a_skill("relationship_mapping")
    async def relationship_mapping_skill(self, entity_data: Dict[str, Any]) -> List[EntityRelationship]:
        """Map relationships between entities"""
        
        relationships = []
        entity_id = entity_data.get('entity_id', str(uuid.uuid4()))
        
        # Generate relationships based on entity attributes
        for related_entity in entity_data.get('related_entities', []):
            relationship = EntityRelationship(
                source_entity=entity_id,
                target_entity=related_entity.get('id'),
                relationship_type=related_entity.get('relationship_type', 'related_to'),
                relationship_strength=related_entity.get('strength', 0.8),
                confidence_score=related_entity.get('confidence', 0.9)
            )
            relationships.append(relationship)
        
        self.processing_stats["relationship_mappings"] += len(relationships)
        return relationships
    
    @a2a_task(
        task_type="ai_entity_preparation",
        description="Complete AI preparation workflow for entities",
        timeout=300,
        retry_attempts=2
    )
    async def prepare_entity_for_ai(self, entity_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete AI preparation workflow"""
        
        entity_id = entity_data.get('entity_id', str(uuid.uuid4()))
        
        try:
            # Stage 1: Semantic enrichment
            semantic_enrichment = await self.execute_skill("semantic_enrichment", entity_data)
            
            # Stage 2: Vectorization
            vector_representation = await self.execute_skill("vectorization", entity_data, semantic_enrichment)
            
            # Stage 3: Relationship mapping
            entity_relationships = await self.execute_skill("relationship_mapping", entity_data)
            
            # Calculate AI readiness score
            ai_readiness_score = self._calculate_ai_readiness_score(entity_data, semantic_enrichment, vector_representation)
            
            # Create AI-ready entity
            ai_ready_entity = AIReadyEntity(
                entity_id=entity_id,
                original_entity=entity_data,
                semantic_enrichment=semantic_enrichment,
                vector_representation=vector_representation,
                entity_relationships=entity_relationships,
                ai_readiness_score=ai_readiness_score,
                quality_metrics=self._calculate_quality_metrics(entity_data, semantic_enrichment, vector_representation),
                processing_metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "agent_version": self.version,
                    "context_id": context_id,
                    "processing_time": time.time()
                }
            )
            
            # Store the AI-ready entity
            self.ai_ready_entities[entity_id] = ai_ready_entity.dict()
            self.processing_stats["total_processed"] += 1
            
            return {
                "ai_preparation_successful": True,
                "entity_id": entity_id,
                "ai_ready_entity": ai_ready_entity.dict(),
                "ai_readiness_score": ai_readiness_score
            }
            
        except Exception as e:
            logger.error(f"AI entity preparation failed: {e}")
            return {
                "ai_preparation_successful": False,
                "entity_id": entity_id,
                "error": str(e)
            }
    
    def _extract_entity_data(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract entity data from message"""
        entity_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                entity_data.update(part.data)
            elif part.kind == "file" and part.file:
                entity_data["file"] = part.file
        
        return entity_data
    
    def _calculate_business_criticality(self, entity_data: Dict[str, Any]) -> float:
        """Calculate business criticality score"""
        base_score = 0.7
        
        # Adjust based on entity type
        if entity_data.get('entity_type') == 'account':
            base_score += 0.2
        elif entity_data.get('entity_type') == 'transaction':
            base_score += 0.15
        
        # Adjust based on volume
        volume = entity_data.get('volume', 0)
        if volume > 10000:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _calculate_strategic_importance(self, entity_data: Dict[str, Any]) -> float:
        """Calculate strategic importance score"""
        base_score = 0.6
        
        # Higher importance for core banking entities
        if entity_data.get('entity_category') in ['core_banking', 'regulatory']:
            base_score += 0.3
        
        return min(base_score, 1.0)
    
    def _calculate_regulatory_complexity(self, entity_data: Dict[str, Any]) -> float:
        """Calculate regulatory complexity score"""
        complexity_factors = len(entity_data.get('compliance_reqs', []))
        return min(complexity_factors * 0.2, 1.0)
    
    def _generate_semantic_description(self, entity_data: Dict[str, Any]) -> str:
        """Generate semantic description for the entity"""
        entity_type = entity_data.get('entity_type', 'financial_entity')
        entity_name = entity_data.get('name', 'unnamed_entity')
        
        return f"A {entity_type} named {entity_name} used in financial data processing and analysis."
    
    def _extract_domain_terminology(self, entity_data: Dict[str, Any]) -> List[str]:
        """Extract relevant domain terminology"""
        terminology = ['financial', 'banking', 'regulatory', 'compliance']
        
        # Add entity-specific terms
        if entity_data.get('entity_type'):
            terminology.append(entity_data['entity_type'])
        
        return terminology
    
    def _generate_synonyms(self, entity_data: Dict[str, Any]) -> List[str]:
        """Generate synonyms and aliases for the entity"""
        synonyms = []
        
        # Add common synonyms based on entity type
        entity_type = entity_data.get('entity_type')
        if entity_type == 'account':
            synonyms.extend(['account', 'banking_account', 'financial_account'])
        elif entity_type == 'transaction':
            synonyms.extend(['transaction', 'payment', 'transfer'])
        
        return synonyms
    
    def _extract_contextual_metadata(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual metadata"""
        return {
            'source_system': entity_data.get('source_system', 'unknown'),
            'data_quality': entity_data.get('quality_score', 0.8),
            'last_updated': entity_data.get('last_updated', datetime.utcnow().isoformat()),
            'entity_lifecycle': entity_data.get('lifecycle_stage', 'active'),
            'enrichment_version': '2.0',
            'confidence_score': 0.95 if self.grok_client else 0.85,
            'enrichment_method': 'grok-ai' if self.grok_client else 'rule-based'
        }
    
    def _prepare_vectorization_text(self, entity_data: Dict[str, Any], semantic_data: SemanticEnrichment) -> str:
        """Prepare text for vectorization"""
        components = [
            semantic_data.semantic_description,
            " ".join(semantic_data.domain_terminology),
            " ".join(semantic_data.synonyms_and_aliases),
            semantic_data.business_context.primary_function
        ]
        
        return " ".join(filter(None, components))
    
    async def _generate_vector_embedding(self, text: str) -> List[float]:
        """Generate vector embedding using sentence transformers or fallback"""
        try:
            # Try to use sentence transformers if available
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model
            model_name = "all-MiniLM-L6-v2"
            if not hasattr(self, '_embedding_model'):
                logger.info(f"Loading embedding model {model_name}...")
                self._embedding_model = SentenceTransformer(model_name)
                self._embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
            
            # Generate embedding
            embedding = self._embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
            
        except ImportError:
            logger.warning("Sentence transformers not available, using hash-based fallback")
            # Fallback to hash-based approach when ML model not available
            import hashlib
            import struct
            
            # Create deterministic embedding based on text hash
            text_hash = hashlib.sha256(text.encode()).digest()
            embedding = []
            
            # Generate 384 dimensions to match all-MiniLM-L6-v2
            for i in range(0, min(len(text_hash), 384//8), 8):
                chunk = text_hash[i:i+8].ljust(8, b'\x00')
                value = struct.unpack('d', chunk)[0]
                # Normalize to [-1, 1] range
                normalized = (value % 2.0) - 1.0
                embedding.append(normalized)
            
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.append(0.0)
            
            return embedding[:384]
    
    def _generate_semantic_tags(self, entity_data: Dict[str, Any], semantic_data: SemanticEnrichment) -> List[str]:
        """Generate semantic tags for the entity"""
        tags = []
        
        # Add entity type tag
        if entity_data.get('entity_type'):
            tags.append(entity_data['entity_type'])
        
        # Add domain tags
        tags.extend(semantic_data.domain_terminology[:3])  # Limit to top 3
        
        # Add business context tags
        tags.append(semantic_data.business_context.primary_function.lower().replace(' ', '_'))
        
        return list(set(tags))  # Remove duplicates
    
    def _calculate_ai_readiness_score(self, entity_data: Dict[str, Any], semantic_data: SemanticEnrichment, vector_data: VectorRepresentation) -> float:
        """Calculate overall AI readiness score"""
        scores = []
        
        # Data completeness score
        required_fields = ['entity_id', 'entity_type', 'name']
        completeness = sum(1 for field in required_fields if entity_data.get(field)) / len(required_fields)
        scores.append(completeness)
        
        # Semantic richness score
        semantic_score = min(len(semantic_data.domain_terminology) * 0.1 + 
                           len(semantic_data.synonyms_and_aliases) * 0.05, 1.0)
        scores.append(semantic_score)
        
        # Vector quality score (based on dimension and non-zero values)
        vector_score = min(vector_data.embedding_dimension / 768.0, 1.0)
        scores.append(vector_score)
        
        # Business relevance score
        business_score = semantic_data.business_context.business_criticality * 0.5 + \
                        semantic_data.business_context.strategic_importance * 0.5
        scores.append(business_score)
        
        return sum(scores) / len(scores)
    
    def _calculate_quality_metrics(self, entity_data: Dict[str, Any], semantic_data: SemanticEnrichment, vector_data: VectorRepresentation) -> Dict[str, float]:
        """Calculate quality metrics"""
        return {
            'data_completeness': sum(1 for v in entity_data.values() if v is not None) / len(entity_data),
            'semantic_richness': len(semantic_data.domain_terminology) * 0.1,
            'vector_quality': 0.95,  # Placeholder
            'business_relevance': semantic_data.business_context.business_criticality,
            'regulatory_compliance': semantic_data.regulatory_context.regulatory_complexity
        }
    
    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            state_file = os.path.join(self.storage_path, "ai_ready_entities.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    self.ai_ready_entities = json.load(f)
                logger.info(f"Loaded {len(self.ai_ready_entities)} AI-ready entities from state")
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save state
            state_file = os.path.join(self.storage_path, "ai_ready_entities.json")
            with open(state_file, 'w') as f:
                json.dump(self.ai_ready_entities, f, default=str, indent=2)
            logger.info(f"Saved {len(self.ai_ready_entities)} AI-ready entities to state")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
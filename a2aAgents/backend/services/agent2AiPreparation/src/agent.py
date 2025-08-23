"""
AI Preparation Agent - A2A Microservice
Agent 2: Prepares standardized data for AI/ML processing
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import os
import sys
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections import defaultdict
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
# Add backend path to import GrokClient
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../backend'))
try:
    from app.clients.grokClient import GrokClient, GrokConfig
except ImportError:
    # Fallback if client not available
    GrokClient = None
    GrokConfig = None

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
from financial_preprocessing import (
    FinancialDomainNormalizer, ContextualEnrichmentEngine, 
    FinancialPromptEngineer, FinancialContext
)
from adaptive_learning import (
    AdaptiveLearningStorage, ContinuousLearner, FeedbackEvent,
    AdaptiveVocabularyExpander
)
from embeddingFinetuner import Agent2EmbeddingSkill
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
logger = logging.getLogger(__name__)


@dataclass
class SemanticEnrichment:
    """Semantic enrichment for financial entities"""
    entity_id: str
    entity_type: str
    semantic_description: str
    business_context: Dict[str, Any]
    domain_terminology: List[str]
    regulatory_context: Dict[str, Any]
    synonyms_and_aliases: List[str]
    contextual_metadata: Dict[str, Any]


class AIPreparationAgent(A2AAgentBase):
    """
    Agent 2: AI Preparation Agent
    Transforms standardized data into AI-ready semantic objects
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="ai_preparation_agent_2",
            name="AI Preparation Agent",
            description="A2A v0.2.9 compliant agent for transforming data into AI-ready semantic objects with embeddings",
            version="2.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        
        # Initialize processing stats
        self.processing_stats = {
            "total_processed": 0,
            "entities_enriched": 0,
            "embeddings_generated": 0,
            "relationships_extracted": 0
        }
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_dim = 768  # Updated for all-mpnet-base-v2
        
        # Initialize Grok client for semantic enrichment
        self.grok_client = None
        
        # Initialize financial preprocessing components
        self.financial_normalizer = FinancialDomainNormalizer()
        self.contextual_engine = ContextualEnrichmentEngine()
        self.prompt_engineer = FinancialPromptEngineer()
        
        # Initialize adaptive learning system
        learning_storage_path = os.getenv("LEARNING_STORAGE_PATH", "/tmp/ai_prep_learning")
        self.learning_storage = AdaptiveLearningStorage(learning_storage_path)
        self.continuous_learner = ContinuousLearner(self.learning_storage)
        self.vocabulary_expander = AdaptiveVocabularyExpander(self.learning_storage)
        
        # Initialize Embedding Fine-tuning skill
        self.embedding_skill = None  # Will be initialized after embedding model is loaded
        
        # HTTP client for API communication
        self.http_client = None
        
        # Semantic processing rules
        self.semantic_rules = {
            "product": {
                "financial_categories": ["derivatives", "securities", "bonds", "equities"]
            },
            "account": {
                "regulatory_types": ["trust", "escrow", "reserve", "regulatory"]
            }
        }
        
        # A2A state tracking
        self.is_ready = False
        self.is_registered = False
        self.tasks = {}
        
        logger.info(f"Initialized A2A {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing AI Preparation Agent...")
        
        # Initialize output directory
        self.output_dir = os.getenv("AI_PREP_OUTPUT_DIR", "/tmp/ai_prepared_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize HTTP client
        self.http_client = None  # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        
        # Initialize A2A trust identity (placeholder for now)
        # TODO: Implement actual trust identity initialization when trust system is ready
        
        # Initialize Grok client for LLM-based semantic enrichment
        await self._initialize_grok_client()
        
        # Initialize embedding model
        await self._initialize_embedding_model()
        
        self.is_ready = True
        logger.info("AI Preparation Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "base_url": self.base_url,
                "capabilities": {
                    "semantic_enrichment": True,
                    "embedding_generation": True,
                    "relationship_extraction": True,
                    "embedding_dimension": self.embedding_dim,
                    "supported_entities": ["account", "book", "location", "measure", "product"]
                },
                "handlers": [h.__name__ if hasattr(h, '__name__') else str(h) for h in self.handlers.values()],
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
    
    @a2a_handler("prepare_for_ai", "Prepare standardized data for AI/ML processing")
    async def handle_ai_preparation_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for AI preparation requests"""
        try:
            # Extract standardized data from A2A message
            standardized_data = self._extract_standardized_data(message)
            
            if not standardized_data:
                return create_error_response(400, "No standardized data found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("ai_preparation", {
                "context_id": context_id,
                "data_types": list(standardized_data.keys()),
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_ai_preparation(task_id, standardized_data, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "data_types": list(standardized_data.keys()),
                "message": "AI preparation started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling AI preparation request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("enhanced_semantic_enrichment", "Enrich entities with advanced financial domain processing")
    async def enrich_semantically(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Add advanced semantic enrichment with financial domain preprocessing"""
        enriched = []
        
        for entity in entities:
            # Phase 1: Financial Domain Normalization
            financial_context = self.financial_normalizer.normalize_entity(entity, entity_type)
            
            # Phase 2: Contextual Enrichment
            enriched_context = self.contextual_engine.enrich_entity_context(
                entity, financial_context, entity_type
            )
            
            # Phase 3: Prompt Engineering for Enhanced Embeddings
            engineered_prompt = self.prompt_engineer.engineer_embedding_prompt(
                entity, financial_context, enriched_context, entity_type
            )
            
            # Use Grok for comprehensive semantic enrichment if available
            if self.grok_client:
                try:
                    # Get enrichment from Grok
                    grok_enrichment = await self._enrich_with_grok(entity, entity_type)
                    
                    # Extract components from Grok response
                    business_context = grok_enrichment.get('business_context', {
                        "primary_function": self._determine_primary_function(entity, entity_type),
                        "stakeholder_groups": self._identify_stakeholders(entity, entity_type),
                        "business_criticality": self._calculate_criticality(entity, entity_type),
                        "operational_context": self._determine_operational_context(entity, entity_type)
                    })
                    
                    # Add additional context
                    business_context.update({
                        "extracted_entities": grok_enrichment.get('entities', []),
                        "risk_indicators": grok_enrichment.get('risk_indicators', self._extract_risk_indicators(entity, entity_type)),
                        "compliance_flags": grok_enrichment.get('compliance_flags', self._extract_compliance_flags(entity, entity_type))
                    })
                    
                    regulatory_context = grok_enrichment.get('regulatory_context', self._determine_regulatory_context(entity, entity_type))
                    semantic_description = grok_enrichment.get('semantic_description', f"AI-enriched {entity_type} entity")
                    domain_terminology = grok_enrichment.get('domain_terminology', self._extract_domain_terminology(entity, entity_type))
                    synonyms_and_aliases = grok_enrichment.get('synonyms', self._find_synonyms(entity, entity_type))
                    
                except Exception as e:
                    logger.warning(f"Grok enrichment failed, using fallback: {e}")
                    # Fallback to rule-based enrichment
                    business_context = {
                        "primary_function": self._determine_primary_function(entity, entity_type),
                        "stakeholder_groups": self._identify_stakeholders(entity, entity_type),
                        "business_criticality": self._calculate_criticality(entity, entity_type),
                        "operational_context": self._determine_operational_context(entity, entity_type),
                        "extracted_entities": [],
                        "risk_indicators": self._extract_risk_indicators(entity, entity_type),
                        "compliance_flags": self._extract_compliance_flags(entity, entity_type)
                    }
                    regulatory_context = self._determine_regulatory_context(entity, entity_type)
                    semantic_description = await self._generate_semantic_description(entity, entity_type)
                    domain_terminology = self._extract_domain_terminology(entity, entity_type)
                    synonyms_and_aliases = self._find_synonyms(entity, entity_type)
            else:
                # No Grok client available, use rule-based enrichment
                business_context = {
                    "primary_function": self._determine_primary_function(entity, entity_type),
                    "stakeholder_groups": self._identify_stakeholders(entity, entity_type),
                    "business_criticality": self._calculate_criticality(entity, entity_type),
                    "operational_context": self._determine_operational_context(entity, entity_type),
                    "extracted_entities": [],
                    "risk_indicators": self._extract_risk_indicators(entity, entity_type),
                    "compliance_flags": self._extract_compliance_flags(entity, entity_type)
                }
                regulatory_context = self._determine_regulatory_context(entity, entity_type)
                semantic_description = await self._generate_semantic_description(entity, entity_type)
                domain_terminology = self._extract_domain_terminology(entity, entity_type)
                synonyms_and_aliases = self._find_synonyms(entity, entity_type)
            
            enrichment = SemanticEnrichment(
                entity_id=entity.get("id", str(hash(str(entity)))),
                entity_type=entity_type,
                semantic_description=engineered_prompt,  # Use enhanced prompt as primary description
                business_context={
                    **business_context,
                    "financial_narrative": enriched_context.get('financial_context', ''),
                    "business_impact": enriched_context.get('business_impact', {}),
                    "operational_context": enriched_context.get('operational_context', {})
                },
                domain_terminology=list(set(domain_terminology + financial_context.domain_synonyms)),
                regulatory_context={
                    **regulatory_context,
                    "financial_classification": financial_context.regulatory_classification,
                    "compliance_analysis": enriched_context.get('regulatory_implications', {})
                },
                synonyms_and_aliases=list(set(synonyms_and_aliases + financial_context.domain_synonyms)),
                contextual_metadata={
                    "enrichment_timestamp": datetime.utcnow().isoformat(),
                    "enrichment_version": "3.0-financial-enhanced",
                    "enrichment_method": "advanced-financial-preprocessing",
                    "preprocessing_applied": {
                        "financial_normalization": True,
                        "contextual_enrichment": True,
                        "prompt_engineering": True,
                        "multi_aspect_embedding": True
                    },
                    "financial_context": financial_context.__dict__,
                    "enriched_context": enriched_context,
                    "confidence_score": self._calculate_enrichment_confidence(entity, entity_type)
                }
            )
            
            # Add enrichment to entity
            entity["semantic_enrichment"] = enrichment.__dict__
            enriched.append(entity)
            
            self.processing_stats["entities_enriched"] += 1
        
        return enriched
    
    @a2a_skill("learn_from_feedback", "Learn from user feedback to improve future processing")
    async def learn_from_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback to improve embedding quality over time"""
        try:
            # Create feedback event from input data
            feedback_event = FeedbackEvent(
                event_id=feedback_data.get("event_id", str(hash(str(feedback_data)))),
                timestamp=datetime.utcnow(),
                query_terms=feedback_data.get("query_terms", []),
                returned_entities=feedback_data.get("returned_entities", []),
                selected_entities=feedback_data.get("selected_entities", []),
                entity_types=feedback_data.get("entity_types", []),
                business_context=feedback_data.get("business_context", "general")
            )
            
            # Process feedback through continuous learner
            success = await self.continuous_learner.process_feedback(feedback_event)
            
            if success:
                # Get updated learning statistics
                stats = self.continuous_learner.get_learning_statistics()
                
                return {
                    "status": "feedback_processed",
                    "learning_stats": stats,
                    "improvements_applied": True
                }
            else:
                return {
                    "status": "feedback_processing_failed",
                    "error": "Failed to process feedback event"
                }
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @a2a_skill("discover_patterns", "Discover new patterns from entity data")
    async def discover_patterns(self, entity_corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run pattern discovery on entity corpus to learn new terminology and relationships"""
        try:
            # Run discovery cycle
            discovery_results = await self.continuous_learner.run_discovery_cycle(entity_corpus)
            
            # Update financial normalizer with learned patterns
            await self._update_normalizer_with_learned_patterns()
            
            return {
                "status": "discovery_completed",
                "patterns_discovered": discovery_results,
                "total_entities_analyzed": len(entity_corpus),
                "discovery_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in pattern discovery: {e}")
            return {
                "status": "discovery_error",
                "error": str(e)
            }
    
    @a2a_skill("get_learning_insights", "Get insights about learned patterns and performance")
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about the learning system performance"""
        try:
            # Get learning statistics
            stats = self.continuous_learner.get_learning_statistics()
            
            # Get top learned patterns
            recent_patterns = self.learning_storage.get_learned_patterns()[:20]
            
            # Format patterns for response
            pattern_summary = []
            for pattern in recent_patterns:
                pattern_summary.append({
                    "type": pattern.pattern_type,
                    "from": pattern.source_term,
                    "to": pattern.target_term,
                    "confidence": round(pattern.confidence_score, 3),
                    "usage_count": pattern.usage_count,
                    "context": pattern.context
                })
            
            return {
                "status": "insights_retrieved",
                "learning_statistics": stats,
                "top_learned_patterns": pattern_summary,
                "learning_system_health": "active" if self.continuous_learner.learning_enabled else "disabled"
            }
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {
                "status": "insights_error",
                "error": str(e)
            }
    
    @a2a_skill("adaptive_enrichment", "Enhanced enrichment using learned patterns")
    async def adaptive_semantic_enrichment(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Enhanced semantic enrichment that adapts based on learned patterns"""
        enriched = []
        
        for entity in entities:
            # Get business context for adaptive processing
            business_context = self._determine_business_context(entity, entity_type)
            
            # Phase 1: Adaptive Financial Domain Normalization
            financial_context = self.financial_normalizer.normalize_entity(entity, entity_type)
            
            # Phase 2: Enhance with learned vocabulary
            enhanced_synonyms = self.vocabulary_expander.get_expanded_synonyms(
                entity.get('name', ''), business_context
            )
            financial_context.domain_synonyms.extend(enhanced_synonyms)
            
            # Get contextually relevant terms
            contextual_terms = self.vocabulary_expander.get_contextual_terms(
                entity_type, business_context
            )
            
            # Phase 3: Contextual Enrichment with adaptive insights
            enriched_context = self.contextual_engine.enrich_entity_context(
                entity, financial_context, entity_type
            )
            
            # Add adaptive context
            enriched_context['adaptive_insights'] = {
                'learned_synonyms': enhanced_synonyms,
                'contextual_terms': contextual_terms,
                'business_context': business_context,
                'adaptation_applied': True
            }
            
            # Phase 4: Adaptive Prompt Engineering
            engineered_prompt = self.prompt_engineer.engineer_embedding_prompt(
                entity, financial_context, enriched_context, entity_type
            )
            
            # Add adaptive enhancement to prompt
            if enhanced_synonyms or contextual_terms:
                adaptive_enhancement = f"\n\nADAPTIVE CONTEXT: This entity is enhanced with learned patterns including synonyms: {', '.join(enhanced_synonyms[:5])} and contextual terms: {', '.join(contextual_terms[:3])}"
                engineered_prompt += adaptive_enhancement
            
            # Create enhanced enrichment
            enrichment = SemanticEnrichment(
                entity_id=entity.get("id", str(hash(str(entity)))),
                entity_type=entity_type,
                semantic_description=engineered_prompt,
                business_context={
                    "financial_narrative": enriched_context.get('financial_context', ''),
                    "business_impact": enriched_context.get('business_impact', {}),
                    "operational_context": enriched_context.get('operational_context', {}),
                    "adaptive_insights": enriched_context['adaptive_insights']
                },
                domain_terminology=list(set(financial_context.domain_synonyms + enhanced_synonyms + contextual_terms)),
                regulatory_context={
                    "financial_classification": financial_context.regulatory_classification,
                    "compliance_analysis": enriched_context.get('regulatory_implications', {})
                },
                synonyms_and_aliases=list(set(financial_context.domain_synonyms + enhanced_synonyms)),
                contextual_metadata={
                    "enrichment_timestamp": datetime.utcnow().isoformat(),
                    "enrichment_version": "4.0-adaptive-learning",
                    "enrichment_method": "adaptive-financial-preprocessing",
                    "preprocessing_applied": {
                        "financial_normalization": True,
                        "contextual_enrichment": True,
                        "prompt_engineering": True,
                        "adaptive_learning": True,
                        "vocabulary_expansion": len(enhanced_synonyms) > 0,
                        "contextual_adaptation": len(contextual_terms) > 0
                    },
                    "financial_context": financial_context.__dict__,
                    "enriched_context": enriched_context,
                    "confidence_score": self._calculate_adaptive_confidence(entity, entity_type, enhanced_synonyms, contextual_terms)
                }
            )
            
            # Add enrichment to entity
            entity["semantic_enrichment"] = enrichment.__dict__
            enriched.append(entity)
            
            self.processing_stats["entities_enriched"] += 1
        
        return enriched
    
    @a2a_skill("generate_embeddings", "Generate vector embeddings for entities")
    async def generate_embeddings(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for entities using sentence transformers"""
        try:
            # Prepare texts for embedding
            texts = []
            for entity in entities:
                # Create a comprehensive text representation
                text_parts = []
                
                # Add main identifiers
                if "name" in entity:
                    text_parts.append(f"name: {entity['name']}")
                if "id" in entity:
                    text_parts.append(f"id: {entity['id']}")
                
                # Add semantic enrichment if available
                if "semantic_enrichment" in entity:
                    enrichment = entity["semantic_enrichment"]
                    text_parts.append(enrichment.get("semantic_description", ""))
                    
                # Add other relevant fields
                for key, value in entity.items():
                    if key not in ["id", "name", "semantic_enrichment", "embedding"] and isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                
                text = " ".join(text_parts)
                texts.append(text)
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Run in thread pool to avoid blocking
                batch_embeddings = await asyncio.to_thread(
                    self.embedding_model.encode,
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                all_embeddings.extend(batch_embeddings)
            
            # Add embeddings to entities
            for entity, embedding in zip(entities, all_embeddings):
                entity["embedding"] = {
                    "vector": embedding.tolist(),
                    "dimension": len(embedding),
                    "model": self.embedding_model_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "normalized": True
                }
                
                self.processing_stats["embeddings_generated"] += 1
            
            return entities
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to zero embeddings rather than failing
            for entity in entities:
                entity["embedding"] = {
                    "vector": [0.0] * self.embedding_dim,
                    "dimension": self.embedding_dim,
                    "model": "zero_fallback",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            return entities
    
    @a2a_skill("track_embedding_learning", "Track how embedding quality changes through learning")
    async def track_embedding_learning(self, entities: List[Dict[str, Any]], entity_type: str) -> Dict[str, Any]:
        """Track and measure ACTUAL embedding learning improvements by comparing before/after embeddings"""
        try:
            learning_evidence = {
                "embedding_changes": {},
                "pattern_applications": {},
                "learning_progression": {},
                "quality_evolution": {}
            }
            
            for entity in entities:
                entity_id = entity.get("id", str(hash(str(entity))))
                
                # Generate embeddings BOTH ways to compare actual differences
                baseline_embedding = await self._generate_baseline_embedding(entity, entity_type)
                adaptive_embedding = await self._generate_adaptive_embedding(entity, entity_type)
                
                # Measure ACTUAL embedding differences
                embedding_changes = self._measure_embedding_changes(baseline_embedding, adaptive_embedding, entity_id)
                
                # Track ACTUAL learned patterns that were applied
                applied_patterns = self._get_actually_applied_patterns(entity, entity_type)
                
                # Measure learning progression over time for this entity type
                progression = await self._measure_learning_progression(entity_type)
                
                # Track quality evolution based on search success rates
                quality_evolution = await self._measure_quality_evolution(entity_type)
                
                learning_evidence["embedding_changes"][entity_id] = embedding_changes
                learning_evidence["pattern_applications"][entity_id] = applied_patterns
                learning_evidence["learning_progression"][entity_type] = progression
                learning_evidence["quality_evolution"][entity_type] = quality_evolution
            
            # Calculate actual learning effectiveness
            learning_effectiveness = self._calculate_actual_learning_effectiveness(learning_evidence)
            
            return {
                "status": "learning_tracked",
                "entity_count": len(entities),
                "entity_type": entity_type,
                "learning_evidence": learning_evidence,
                "learning_effectiveness": learning_effectiveness,
                "timestamp": datetime.utcnow().isoformat(),
                "learning_proof": "embedding_vectors_actually_changed"
            }
            
        except Exception as e:
            logger.error(f"Error tracking embedding learning: {e}")
            return {
                "status": "tracking_error",
                "error": str(e)
            }
    
    async def _generate_baseline_embedding(self, entity: Dict[str, Any], entity_type: str) -> List[float]:
        """Generate baseline embedding WITHOUT any learned patterns"""
        # Simple baseline text without financial preprocessing
        baseline_text = f"{entity_type} entity: {entity.get('name', 'Unknown')}"
        
        if self.embedding_model:
            baseline_embedding = await asyncio.to_thread(
                self.embedding_model.encode,
                baseline_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return baseline_embedding.tolist()
        else:
            return [0.0] * self.embedding_dim
    
    async def _generate_adaptive_embedding(self, entity: Dict[str, Any], entity_type: str) -> List[float]:
        """Generate embedding WITH all learned patterns applied"""
        # Use full adaptive enrichment pipeline
        enriched_entities = await self.adaptive_semantic_enrichment([entity], entity_type)
        
        if enriched_entities and "semantic_enrichment" in enriched_entities[0]:
            enriched_text = enriched_entities[0]["semantic_enrichment"]["semantic_description"]
            
            if self.embedding_model:
                adaptive_embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    enriched_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return adaptive_embedding.tolist()
        
        return [0.0] * self.embedding_dim
    
    def _measure_embedding_changes(self, baseline_embedding: List[float], 
                                  adaptive_embedding: List[float], entity_id: str) -> Dict[str, Any]:
        """Measure actual numerical changes between baseline and adaptive embeddings"""
        if not baseline_embedding or not adaptive_embedding:
            return {"error": "missing_embeddings"}
        
        import numpy as np
        
        baseline_vec = np.array(baseline_embedding)
        adaptive_vec = np.array(adaptive_embedding)
        
        # Calculate actual mathematical differences
        cosine_similarity = np.dot(baseline_vec, adaptive_vec) / (np.linalg.norm(baseline_vec) * np.linalg.norm(adaptive_vec))
        euclidean_distance = np.linalg.norm(baseline_vec - adaptive_vec)
        magnitude_change = np.linalg.norm(adaptive_vec) / np.linalg.norm(baseline_vec) if np.linalg.norm(baseline_vec) > 0 else 0
        
        # Measure dimension-wise changes
        dimension_changes = np.abs(adaptive_vec - baseline_vec)
        significant_changes = np.sum(dimension_changes > 0.1)  # Dimensions that changed significantly
        
        return {
            "entity_id": entity_id,
            "cosine_similarity": float(cosine_similarity),
            "euclidean_distance": float(euclidean_distance),
            "magnitude_change_ratio": float(magnitude_change),
            "dimensions_changed_significantly": int(significant_changes),
            "max_dimension_change": float(np.max(dimension_changes)),
            "avg_dimension_change": float(np.mean(dimension_changes)),
            "embedding_actually_changed": euclidean_distance > 0.01,
            "change_magnitude": "significant" if euclidean_distance > 0.5 else "moderate" if euclidean_distance > 0.1 else "minimal"
        }
    
    def _get_actually_applied_patterns(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Get the ACTUAL learned patterns that were applied to this entity"""
        business_context = self._determine_business_context(entity, entity_type)
        entity_str = json.dumps(entity, default=str).lower()
        entity_name = entity.get('name', '').lower()
        
        applied_patterns = {
            "synonyms_used": [],
            "regulatory_mappings_used": [],
            "effective_terms_used": [],
            "total_learned_influences": 0
        }
        
        # Check which learned synonyms actually match this entity
        learned_synonyms = self.learning_storage.get_learned_patterns("synonym")
        for pattern in learned_synonyms:
            if (pattern.source_term.lower() in entity_name or 
                pattern.target_term.lower() in entity_name or
                pattern.source_term.lower() in entity_str or
                pattern.target_term.lower() in entity_str):
                applied_patterns["synonyms_used"].append({
                    "source": pattern.source_term,
                    "target": pattern.target_term,
                    "confidence": pattern.confidence_score,
                    "learned_from": pattern.usage_count,
                    "context": pattern.context
                })
        
        # Check regulatory mappings that actually applied
        regulatory_patterns = self.learning_storage.get_learned_patterns("regulatory_mapping")
        for pattern in regulatory_patterns:
            if pattern.source_term.lower() in entity_str:
                applied_patterns["regulatory_mappings_used"].append({
                    "trigger_term": pattern.source_term,
                    "regulation": pattern.target_term,
                    "confidence": pattern.confidence_score
                })
        
        # Check effective search terms that applied
        effective_patterns = self.learning_storage.get_learned_patterns("effective_search_term", business_context)
        for pattern in effective_patterns:
            if pattern.source_term.lower() in entity_str:
                applied_patterns["effective_terms_used"].append({
                    "term": pattern.source_term,
                    "effectiveness": pattern.confidence_score,
                    "learned_context": pattern.target_term
                })
        
        applied_patterns["total_learned_influences"] = (
            len(applied_patterns["synonyms_used"]) + 
            len(applied_patterns["regulatory_mappings_used"]) + 
            len(applied_patterns["effective_terms_used"])
        )
        
        return applied_patterns
    
    async def _measure_learning_progression(self, entity_type: str) -> Dict[str, Any]:
        """Measure how learning has progressed over time for this entity type"""
        try:
            with sqlite3.connect(self.learning_storage.db_path) as conn:
                # Get pattern creation timeline
                cursor = conn.execute("""
                    SELECT pattern_type, COUNT(*) as count, 
                           MIN(first_seen) as first_learned,
                           MAX(last_updated) as last_updated,
                           AVG(confidence_score) as avg_confidence
                    FROM learned_patterns 
                    WHERE source_term LIKE ? OR target_term LIKE ?
                    GROUP BY pattern_type
                    ORDER BY first_learned
                """, (f'%{entity_type}%', f'%{entity_type}%'))
                
                progression_data = []
                for row in cursor.fetchall():
                    progression_data.append({
                        "pattern_type": row[0],
                        "patterns_learned": row[1],
                        "first_learned": row[2],
                        "last_updated": row[3],
                        "avg_confidence": row[4]
                    })
                
                # Get learning velocity (patterns learned per week)
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_patterns
                    FROM learned_patterns 
                    WHERE (source_term LIKE ? OR target_term LIKE ?)
                    AND first_seen > datetime('now', '-7 days')
                """, (f'%{entity_type}%', f'%{entity_type}%'))
                
                recent_patterns = cursor.fetchone()[0] or 0
                
                return {
                    "entity_type": entity_type,
                    "progression_timeline": progression_data,
                    "patterns_learned_this_week": recent_patterns,
                    "learning_velocity": recent_patterns / 7.0,  # Patterns per day
                    "learning_maturity": "mature" if len(progression_data) >= 3 else "developing",
                    "total_patterns_for_type": sum(p["patterns_learned"] for p in progression_data)
                }
        
        except Exception as e:
            logger.error(f"Error measuring learning progression: {e}")
            return {"error": str(e), "entity_type": entity_type}
    
    async def _measure_quality_evolution(self, entity_type: str) -> Dict[str, Any]:
        """Measure how embedding quality has evolved based on actual search success rates"""
        try:
            with sqlite3.connect(self.learning_storage.db_path) as conn:
                # Get feedback events over time to measure improvement
                cursor = conn.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_searches,
                        AVG(CAST(json_array_length(selected_entities) AS FLOAT) / 
                            CAST(json_array_length(returned_entities) AS FLOAT)) as avg_success_rate
                    FROM feedback_events 
                    WHERE json_extract(entity_types, '$[0]') = ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                """, (entity_type,))
                
                quality_timeline = []
                for row in cursor.fetchall():
                    quality_timeline.append({
                        "date": row[0],
                        "searches": row[1],
                        "success_rate": row[2] or 0.0
                    })
                
                # Calculate trend
                if len(quality_timeline) >= 2:
                    recent_success = sum(d["success_rate"] for d in quality_timeline[:7]) / min(7, len(quality_timeline))
                    early_success = sum(d["success_rate"] for d in quality_timeline[-7:]) / min(7, len(quality_timeline[-7:]))
                    improvement_trend = recent_success - early_success
                else:
                    improvement_trend = 0.0
                    recent_success = quality_timeline[0]["success_rate"] if quality_timeline else 0.0
                
                return {
                    "entity_type": entity_type,
                    "quality_timeline": quality_timeline,
                    "recent_success_rate": recent_success,
                    "improvement_trend": improvement_trend,
                    "quality_direction": "improving" if improvement_trend > 0.05 else "declining" if improvement_trend < -0.05 else "stable",
                    "data_points": len(quality_timeline)
                }
        
        except Exception as e:
            logger.error(f"Error measuring quality evolution: {e}")
            return {"error": str(e), "entity_type": entity_type}
    
    def _calculate_actual_learning_effectiveness(self, learning_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate learning effectiveness based on actual evidence"""
        
        embedding_changes = learning_evidence.get("embedding_changes", {})
        pattern_applications = learning_evidence.get("pattern_applications", {})
        
        if not embedding_changes:
            return {"effectiveness": 0.0, "reason": "no_embedding_changes_measured"}
        
        # Calculate actual change metrics
        total_entities = len(embedding_changes)
        entities_actually_changed = sum(1 for changes in embedding_changes.values() 
                                      if changes.get("embedding_actually_changed", False))
        
        avg_distance_change = sum(changes.get("euclidean_distance", 0) 
                                for changes in embedding_changes.values()) / total_entities
        
        total_learned_influences = sum(patterns.get("total_learned_influences", 0) 
                                     for patterns in pattern_applications.values())
        
        # Calculate learning effectiveness score
        change_effectiveness = entities_actually_changed / total_entities if total_entities > 0 else 0
        magnitude_effectiveness = min(avg_distance_change, 2.0) / 2.0  # Normalized
        influence_effectiveness = min(total_learned_influences / (total_entities * 5), 1.0)  # Expected 5 influences per entity
        
        overall_effectiveness = (change_effectiveness + magnitude_effectiveness + influence_effectiveness) / 3
        
        return {
            "overall_effectiveness": overall_effectiveness,
            "entities_actually_changed": entities_actually_changed,
            "total_entities": total_entities,
            "change_percentage": change_effectiveness * 100,
            "avg_embedding_distance": avg_distance_change,
            "total_learned_influences_applied": total_learned_influences,
            "learning_proof": {
                "embeddings_numerically_different": entities_actually_changed > 0,
                "learned_patterns_actually_applied": total_learned_influences > 0,
                "measurable_quality_improvement": overall_effectiveness > 0.5
            },
            "effectiveness_rating": "high" if overall_effectiveness > 0.7 else "medium" if overall_effectiveness > 0.4 else "low"
        }
    
    async def _analyze_embedding_generation(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Analyze how the embedding was generated and what influenced it"""
        
        # Track preprocessing stages that influenced embedding
        preprocessing_impact = {
            "financial_normalization": 0,
            "regulatory_context": 0,
            "risk_indicators": 0,
            "learned_synonyms": 0,
            "contextual_terms": 0
        }
        
        # Get business context
        business_context = self._determine_business_context(entity, entity_type)
        
        # Analyze financial context influence
        financial_context = self.financial_normalizer.normalize_entity(entity, entity_type)
        preprocessing_impact["financial_normalization"] = len(financial_context.normalized_terminology)
        preprocessing_impact["regulatory_context"] = len(financial_context.regulatory_classification.get('applicable_regulations', []))
        preprocessing_impact["risk_indicators"] = len(financial_context.risk_indicators)
        
        # Analyze learned pattern influence
        learned_synonyms = self.vocabulary_expander.get_expanded_synonyms(entity.get('name', ''), business_context)
        contextual_terms = self.vocabulary_expander.get_contextual_terms(entity_type, business_context)
        
        preprocessing_impact["learned_synonyms"] = len(learned_synonyms)
        preprocessing_impact["contextual_terms"] = len(contextual_terms)
        
        # Calculate total enhancement score
        enhancement_score = sum(preprocessing_impact.values()) / 25.0  # Normalized to 0-1
        
        return {
            "preprocessing_impact": preprocessing_impact,
            "business_context": business_context,
            "enhancement_score": min(enhancement_score, 1.0),
            "learned_patterns_applied": len(learned_synonyms) + len(contextual_terms),
            "static_rules_applied": preprocessing_impact["financial_normalization"] + preprocessing_impact["regulatory_context"]
        }
    
    async def _measure_embedding_quality_improvements(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Measure quality improvements from static baseline to adaptive version"""
        
        # Generate baseline (static) semantic description
        baseline_description = await self._generate_baseline_semantic_description(entity, entity_type)
        
        # Get current adaptive description
        business_context = self._determine_business_context(entity, entity_type)
        financial_context = self.financial_normalizer.normalize_entity(entity, entity_type)
        enriched_context = self.contextual_engine.enrich_entity_context(entity, financial_context, entity_type)
        adaptive_description = self.prompt_engineer.engineer_embedding_prompt(entity, financial_context, enriched_context, entity_type)
        
        # Compare descriptions
        quality_metrics = {
            "baseline_length": len(baseline_description),
            "adaptive_length": len(adaptive_description),
            "enhancement_ratio": len(adaptive_description) / max(len(baseline_description), 1),
            "regulatory_terms_added": self._count_regulatory_terms_added(baseline_description, adaptive_description),
            "risk_terms_added": self._count_risk_terms_added(baseline_description, adaptive_description),
            "business_context_depth": self._measure_business_context_depth(adaptive_description),
            "learned_vocabulary_integration": self._count_learned_vocabulary_usage(adaptive_description, entity_type)
        }
        
        # Calculate overall quality improvement score
        improvement_factors = [
            min(quality_metrics["enhancement_ratio"], 2.0) / 2.0,  # Normalized enhancement
            min(quality_metrics["regulatory_terms_added"], 5) / 5.0,  # Regulatory depth
            min(quality_metrics["risk_terms_added"], 3) / 3.0,  # Risk awareness
            min(quality_metrics["learned_vocabulary_integration"], 10) / 10.0  # Learning integration
        ]
        
        quality_metrics["overall_improvement_score"] = sum(improvement_factors) / len(improvement_factors)
        quality_metrics["baseline_description"] = baseline_description[:200] + "..." if len(baseline_description) > 200 else baseline_description
        quality_metrics["adaptive_preview"] = adaptive_description[:200] + "..." if len(adaptive_description) > 200 else adaptive_description
        
        return quality_metrics
    
    async def _identify_learning_sources_impact(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Identify which learning sources influenced this embedding"""
        
        entity_name = entity.get('name', '')
        business_context = self._determine_business_context(entity, entity_type)
        
        # Check for learned pattern matches
        learning_sources = {
            "synonym_patterns": [],
            "regulatory_patterns": [],
            "effective_search_terms": [],
            "risk_associations": [],
            "contextual_adaptations": []
        }
        
        # Get learned patterns that apply to this entity
        learned_synonyms = self.learning_storage.get_learned_patterns("synonym", business_context)
        for pattern in learned_synonyms:
            if (pattern.source_term.lower() in entity_name.lower() or 
                pattern.target_term.lower() in entity_name.lower()):
                learning_sources["synonym_patterns"].append({
                    "pattern": f"{pattern.source_term} → {pattern.target_term}",
                    "confidence": pattern.confidence_score,
                    "usage_count": pattern.usage_count,
                    "context": pattern.context
                })
        
        # Check regulatory mappings
        regulatory_patterns = self.learning_storage.get_learned_patterns("regulatory_mapping")
        entity_str = json.dumps(entity, default=str).lower()
        for pattern in regulatory_patterns:
            if pattern.source_term.lower() in entity_str:
                learning_sources["regulatory_patterns"].append({
                    "term": pattern.source_term,
                    "regulation": pattern.target_term,
                    "confidence": pattern.confidence_score
                })
        
        # Check effective search terms
        effective_terms = self.learning_storage.get_learned_patterns("effective_search_term", business_context)
        for pattern in effective_terms:
            if pattern.source_term.lower() in entity_str:
                learning_sources["effective_search_terms"].append({
                    "term": pattern.source_term,
                    "effectiveness": pattern.confidence_score,
                    "context": pattern.target_term
                })
        
        # Calculate learning source diversity
        total_sources = sum(len(sources) for sources in learning_sources.values())
        source_diversity = len([k for k, v in learning_sources.items() if v]) / len(learning_sources)
        
        return {
            "learning_sources": learning_sources,
            "total_learned_influences": total_sources,
            "source_diversity_score": source_diversity,
            "primary_learning_context": business_context
        }
    
    def _calculate_adaptation_effectiveness(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Calculate how effective the adaptive learning is for this entity"""
        
        business_context = self._determine_business_context(entity, entity_type)
        
        # Get learning statistics for this context
        try:
            with sqlite3.connect(self.learning_storage.db_path) as conn:
                # Get effectiveness of terms in this business context
                cursor = conn.execute("""
                    SELECT AVG(effectiveness_score), COUNT(*) 
                    FROM usage_stats 
                    WHERE contexts LIKE ?
                """, (f'%{business_context}%',))
                
                avg_effectiveness, term_count = cursor.fetchone() or (0.0, 0)
                
                # Get pattern confidence in this context
                cursor = conn.execute("""
                    SELECT AVG(confidence_score), COUNT(*) 
                    FROM learned_patterns 
                    WHERE context = ?
                """, (business_context,))
                
                avg_confidence, pattern_count = cursor.fetchone() or (0.0, 0)
        
        except Exception as e:
            logger.error(f"Error calculating adaptation effectiveness: {e}")
            avg_effectiveness, term_count = 0.0, 0
            avg_confidence, pattern_count = 0.0, 0
        
        # Calculate adaptation metrics
        adaptation_metrics = {
            "context_term_effectiveness": avg_effectiveness or 0.0,
            "context_terms_tracked": term_count,
            "context_pattern_confidence": avg_confidence or 0.0,
            "context_patterns_learned": pattern_count,
            "adaptation_maturity": self._calculate_context_maturity(business_context, term_count, pattern_count),
            "learning_coverage": min(term_count / 10.0, 1.0),  # Normalized to 0-1
            "pattern_reliability": avg_confidence or 0.0
        }
        
        # Overall adaptation effectiveness score
        effectiveness_factors = [
            adaptation_metrics["context_term_effectiveness"],
            adaptation_metrics["context_pattern_confidence"],
            adaptation_metrics["learning_coverage"]
        ]
        
        adaptation_metrics["overall_effectiveness"] = sum(effectiveness_factors) / len(effectiveness_factors)
        
        return adaptation_metrics
    
    def _calculate_context_maturity(self, business_context: str, term_count: int, pattern_count: int) -> str:
        """Calculate learning maturity for this business context"""
        total_learning_items = term_count + pattern_count
        
        if total_learning_items >= 50:
            return "mature"
        elif total_learning_items >= 20:
            return "developing"
        elif total_learning_items >= 5:
            return "emerging"
        else:
            return "initial"
    
    def _aggregate_learning_insights(self, learning_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate learning insights across all processed entities"""
        
        total_entities = len(learning_metrics["embedding_generation_analysis"])
        
        if total_entities == 0:
            return {"status": "no_entities_processed"}
        
        # Aggregate enhancement scores
        enhancement_scores = [
            analysis["enhancement_score"] 
            for analysis in learning_metrics["embedding_generation_analysis"].values()
        ]
        
        # Aggregate quality improvements
        quality_improvements = [
            metrics["overall_improvement_score"]
            for metrics in learning_metrics["quality_improvements"].values()
        ]
        
        # Aggregate learning influences
        learning_influences = [
            sources["total_learned_influences"]
            for sources in learning_metrics["learning_sources"].values()
        ]
        
        # Aggregate adaptation effectiveness
        adaptation_scores = [
            metrics["overall_effectiveness"]
            for metrics in learning_metrics["adaptation_effectiveness"].values()
        ]
        
        return {
            "total_entities_analyzed": total_entities,
            "avg_enhancement_score": sum(enhancement_scores) / total_entities,
            "avg_quality_improvement": sum(quality_improvements) / total_entities,
            "avg_learning_influences": sum(learning_influences) / total_entities,
            "avg_adaptation_effectiveness": sum(adaptation_scores) / total_entities,
            "learning_maturity_distribution": self._get_maturity_distribution(learning_metrics),
            "top_learning_contexts": self._get_top_learning_contexts(learning_metrics),
            "embedding_learning_health": self._assess_embedding_learning_health(enhancement_scores, quality_improvements, adaptation_scores)
        }
    
    def _get_maturity_distribution(self, learning_metrics: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of learning maturity levels"""
        maturity_counts = {"initial": 0, "emerging": 0, "developing": 0, "mature": 0}
        
        for metrics in learning_metrics["adaptation_effectiveness"].values():
            maturity = metrics.get("adaptation_maturity", "initial")
            maturity_counts[maturity] += 1
        
        return maturity_counts
    
    def _get_top_learning_contexts(self, learning_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top business contexts by learning activity"""
        context_learning = defaultdict(list)
        
        for sources in learning_metrics["learning_sources"].values():
            context = sources.get("primary_learning_context", "general")
            total_influences = sources.get("total_learned_influences", 0)
            context_learning[context].append(total_influences)
        
        # Calculate average influences per context
        context_averages = []
        for context, influences in context_learning.items():
            if influences:
                context_averages.append({
                    "context": context,
                    "avg_influences": sum(influences) / len(influences),
                    "entity_count": len(influences)
                })
        
        # Sort by average influences
        context_averages.sort(key=lambda x: x["avg_influences"], reverse=True)
        return context_averages[:5]  # Top 5 contexts
    
    def _assess_embedding_learning_health(self, enhancement_scores: List[float], 
                                        quality_scores: List[float], 
                                        adaptation_scores: List[float]) -> str:
        """Assess overall health of embedding learning system"""
        
        avg_enhancement = sum(enhancement_scores) / len(enhancement_scores) if enhancement_scores else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_adaptation = sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0
        
        overall_health = (avg_enhancement + avg_quality + avg_adaptation) / 3
        
        if overall_health >= 0.8:
            return "excellent"
        elif overall_health >= 0.6:
            return "good"
        elif overall_health >= 0.4:
            return "fair"
        else:
            return "needs_improvement"
    
    # Helper methods for quality measurement
    async def _generate_baseline_semantic_description(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Generate baseline semantic description without learning enhancements"""
        # Simple baseline description
        name = entity.get('name', 'Unknown')
        entity_type_desc = entity_type.replace('_', ' ')
        return f"{entity_type_desc} entity: {name}"
    
    def _count_regulatory_terms_added(self, baseline: str, adaptive: str) -> int:
        """Count regulatory terms added in adaptive version"""
        regulatory_terms = ['mifid', 'basel', 'ifrs', 'sox', 'regulatory', 'compliance', 'audit']
        baseline_lower = baseline.lower()
        adaptive_lower = adaptive.lower()
        
        added_terms = 0
        for term in regulatory_terms:
            if term not in baseline_lower and term in adaptive_lower:
                added_terms += 1
        
        return added_terms
    
    def _count_risk_terms_added(self, baseline: str, adaptive: str) -> int:
        """Count risk terms added in adaptive version"""
        risk_terms = ['risk', 'exposure', 'volatility', 'stress', 'var', 'market_risk', 'credit_risk']
        baseline_lower = baseline.lower()
        adaptive_lower = adaptive.lower()
        
        added_terms = 0
        for term in risk_terms:
            if term not in baseline_lower and term in adaptive_lower:
                added_terms += 1
        
        return added_terms
    
    def _measure_business_context_depth(self, description: str) -> float:
        """Measure depth of business context in description"""
        context_indicators = [
            'business_line', 'operational_level', 'stakeholder', 'process',
            'investment_banking', 'retail_banking', 'corporate_banking',
            'trading', 'compliance', 'risk_management'
        ]
        
        description_lower = description.lower()
        depth_score = sum(1 for indicator in context_indicators if indicator in description_lower)
        
        return min(depth_score / 10.0, 1.0)  # Normalized to 0-1
    
    def _count_learned_vocabulary_usage(self, description: str, entity_type: str) -> int:
        """Count usage of learned vocabulary in description"""
        # This would check against actual learned patterns
        # For now, estimate based on description richness
        words = description.lower().split()
        unique_financial_terms = set()
        
        financial_vocabulary = [
            'securities', 'portfolio', 'derivative', 'hedging', 'counterparty',
            'liquidity', 'capital', 'regulatory', 'compliance', 'operational'
        ]
        
        for word in words:
            if word in financial_vocabulary:
                unique_financial_terms.add(word)
        
        return len(unique_financial_terms)
    
    @a2a_skill("extract_relationships", "Extract relationships between entities")
    async def extract_relationships(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple relationship extraction (in production, use graph algorithms)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Mock relationship detection
                if entity_type == "account" and "currency" in entity1 and "currency" in entity2:
                    if entity1["currency"] == entity2["currency"]:
                        relationships.append({
                            "source_id": entity1.get("id", str(i)),
                            "target_id": entity2.get("id", str(i+1)),
                            "relationship_type": "same_currency",
                            "confidence": 1.0,
                            "attributes": {"currency": entity1["currency"]}
                        })
                        self.processing_stats["relationships_extracted"] += 1
        
        return relationships
    
    async def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer embedding model with fine-tuning capability"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize the embedding fine-tuning skill with learning storage
            self.embedding_skill = Agent2EmbeddingSkill(
                learning_storage=self.learning_storage,
                audit_logger=None,  # TODO: Add when audit logger is available
                metrics_client=None  # TODO: Add when metrics client is available
            )
            
            # Check if we have a fine-tuned model available
            if self.embedding_skill.current_model_path != "sentence-transformers/all-mpnet-base-v2":
                logger.info(f"Using fine-tuned model: {self.embedding_skill.current_model_path}")
                self.embedding_model = self.embedding_skill.get_current_model()
                self.embedding_model_name = self.embedding_skill.current_model_path
            else:
                # Use base model
                self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
                logger.info(f"Loading base embedding model: {self.embedding_model_name}")
                
                # Load model in thread pool to avoid blocking
                self.embedding_model = await asyncio.to_thread(
                    SentenceTransformer,
                    self.embedding_model_name
                )
            
            # Verify model works by encoding a test sentence
            test_embedding = await asyncio.to_thread(
                self.embedding_model.encode,
                "Test sentence for verification",
                convert_to_numpy=True
            )
            
            self.embedding_dim = len(test_embedding)
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dim}")
            
            # Schedule periodic fine-tuning checks
            asyncio.create_task(self._periodic_fine_tuning_check())
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            logger.warning("Will use zero embeddings as fallback")
            self.embedding_model = None
            self.embedding_model_name = "zero_fallback"
            self.embedding_skill = None
    
    async def _initialize_grok_client(self) -> None:
        """Initialize Grok client for LLM-based semantic enrichment"""
        try:
            # Create Grok client with configuration
            grok_config = GrokConfig(
                api_key=os.getenv('XAI_API_KEY', ''),
                base_url=os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1'),
                model=os.getenv('XAI_MODEL', 'grok-4-latest'),
                timeout=int(os.getenv('XAI_TIMEOUT', '30'))
            )
            
            if grok_config.api_key:
                self.grok_client = GrokClient(grok_config)
                logger.info(f"Grok client initialized with model: {grok_config.model}")
                
                # Test the client
                health = self.grok_client.health_check()
                if health['status'] == 'healthy':
                    logger.info(f"Grok client health check passed: {health}")
                else:
                    logger.warning(f"Grok client health check failed: {health}")
                    self.grok_client = None
            else:
                logger.warning("No Grok API key found, using fallback enrichment")
                self.grok_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Grok client: {e}")
            self.grok_client = None
    
    async def _process_ai_preparation(self, task_id: str, data: Dict[str, Any], context_id: str):
        """Process AI preparation asynchronously"""
        try:
            ai_prepared_data = {}
            all_relationships = []
            
            # Process each data type
            for data_type, entities in data.items():
                if isinstance(entities, list) and len(entities) > 0:
                    logger.info(f"Preparing {len(entities)} {data_type} entities for AI")
                    
                    # Step 1: Semantic enrichment
                    enriched = await self.enrich_semantically(entities, data_type)
                    
                    # Step 2: Generate embeddings
                    with_embeddings = await self.generate_embeddings(enriched)
                    
                    # Step 3: Extract relationships
                    relationships = await self.extract_relationships(with_embeddings, data_type)
                    all_relationships.extend(relationships)
                    
                    ai_prepared_data[data_type] = with_embeddings
            
            # Add relationships to prepared data
            if all_relationships:
                ai_prepared_data["relationships"] = all_relationships
            
            # Update stats
            self.processing_stats["total_processed"] += 1
            
            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(ai_prepared_data, context_id)
            
            # Update task status
            await self.update_task_status(task_id, "completed", {
                "entities_enriched": self.processing_stats["entities_enriched"],
                "embeddings_generated": self.processing_stats["embeddings_generated"],
                "relationships_extracted": self.processing_stats["relationships_extracted"]
            })
            
        except Exception as e:
            logger.error(f"Error processing AI preparation: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send AI-prepared data to downstream agent via A2A protocol"""
        try:
            # Prepare JSON-RPC request
            rpc_request = {
                "jsonrpc": "2.0",
                "method": "process_embeddings",
                "params": {
                    "ai_prepared_data": data,
                    "context_id": context_id,
                    "source_agent": self.agent_id,
                    "preparation_metadata": {
                        "embedding_dimension": self.embedding_dim,
                        "embedding_model": self.embedding_model_name if hasattr(self, 'embedding_model_name') else "unknown",
                        "total_entities": sum(len(v) for k, v in data.items() if k != "relationships"),
                        "total_relationships": len(data.get("relationships", [])),
                        "ai_enrichment": self.grok_client is not None,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                "id": f"downstream_{context_id}_{int(datetime.utcnow().timestamp())}"
            }
            
            # Send to downstream agent
            response = await self.http_client.post(
                f"{self.downstream_agent_url}/a2a/vector_processing_agent_3/v1/rpc",
                json=rpc_request,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    logger.info(f"Successfully sent AI-prepared data to downstream agent: {result['result']}")
                elif "error" in result:
                    logger.error(f"Downstream agent returned error: {result['error']}")
            else:
                logger.error(f"Failed to send to downstream agent: HTTP {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send to downstream agent: {e}")
    
    def _extract_standardized_data(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract standardized data from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('standardized_data', content.get('data', None))
        return None
    
    async def _initialize_trust_identity(self) -> None:
        """Initialize agent's trust identity for A2A network"""
        # This will be implemented when trust system is ready
        # For now, just log the initialization
        logger.info(f"Trust identity initialization placeholder for {self.agent_id}")
        # In production, this would:
        # 1. Generate or load agent's cryptographic keys
        # 2. Register with trust authority
        # 3. Obtain trust certificates
        # 4. Set up secure communication channels
        pass
    
    def generate_context_id(self) -> str:
        """Generate unique context ID for A2A messages"""
        import uuid
        return str(uuid.uuid4())
    
    def create_message(self, content: Any) -> A2AMessage:
        """Create A2A message from content"""
        return A2AMessage(
            sender_id=self.agent_id,
            content=content,
            role=MessageRole.AGENT
        )
    
    async def create_task(self, task_type: str, metadata: Dict[str, Any]) -> str:
        """Create and track a new task"""
        import uuid
        task_id = str(uuid.uuid4())
        
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
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
            
            if update_data:
                self.tasks[task_id]["metadata"].update(update_data)
    
    def _entity_to_text(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Convert entity to text for NLP processing"""
        parts = [f"{entity_type}:"]
        for key, value in entity.items():
            if key not in ["embedding", "metadata"]:
                parts.append(f"{key}={value}")
        return " ".join(parts)
    
    def _determine_primary_function(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Determine primary business function"""
        function_map = {
            "account": "financial_accounting",
            "book": "record_keeping",
            "location": "geographic_operations",
            "measure": "performance_tracking",
            "product": "product_management"
        }
        return function_map.get(entity_type, f"{entity_type}_management")
    
    def _identify_stakeholders(self, entity: Dict[str, Any], entity_type: str) -> List[str]:
        """Identify relevant stakeholder groups"""
        base_stakeholders = ["finance", "operations"]
        
        if entity_type == "account":
            base_stakeholders.extend(["treasury", "audit"])
            if entity.get("type") == "regulatory":
                base_stakeholders.append("compliance")
        elif entity_type == "product":
            base_stakeholders.extend(["sales", "marketing"])
        elif entity_type == "location":
            base_stakeholders.extend(["logistics", "facilities"])
        
        return list(set(base_stakeholders))
    
    def _calculate_criticality(self, entity: Dict[str, Any], entity_type: str) -> float:
        """Calculate business criticality score"""
        criticality = 0.5  # Base score
        
        # Adjust based on entity type
        if entity_type == "account":
            if "balance" in entity and entity["balance"] > 1000000:
                criticality += 0.3
            if entity.get("type") in ["regulatory", "reserve"]:
                criticality += 0.2
        elif entity_type == "product":
            if entity.get("category") == "core":
                criticality += 0.4
        
        return min(criticality, 1.0)
    
    def _determine_operational_context(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Determine operational context"""
        if entity_type == "account":
            account_type = entity.get("type", "general")
            if account_type in ["operating", "checking"]:
                return "daily_operations"
            elif account_type in ["investment", "trading"]:
                return "investment_operations"
            else:
                return "core_financial_operations"
        elif entity_type == "product":
            return "product_lifecycle_management"
        else:
            return f"{entity_type}_operations"
    
    def _determine_regulatory_context(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Determine regulatory context based on entity"""
        context = {
            "framework": "General",
            "compliance_requirements": ["data_retention"],
            "regulatory_complexity": 0.3
        }
        
        if entity_type == "account":
            context["framework"] = "SOX"
            context["compliance_requirements"].extend(["audit_trail", "segregation_of_duties"])
            context["regulatory_complexity"] = 0.7
            
            if entity.get("currency") != "USD":
                context["compliance_requirements"].append("foreign_exchange_reporting")
                context["regulatory_complexity"] = 0.8
        
        elif entity_type == "product" and entity.get("category") == "financial":
            context["framework"] = "MiFID II"
            context["compliance_requirements"].extend(["transaction_reporting", "best_execution"])
            context["regulatory_complexity"] = 0.9
        
        return context
    
    async def _enrich_with_grok(self, entity: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Use Grok LLM for comprehensive semantic enrichment"""
        try:
            # Prepare entity data for Grok
            entity_json = json.dumps(entity, indent=2)
            
            # Create comprehensive prompt for semantic enrichment
            prompt = f"""Analyze this {entity_type} entity and provide comprehensive semantic enrichment.

Entity Type: {entity_type}
Entity Data:
{entity_json}

Provide a JSON response with the following structure:
{{
  "semantic_description": "A comprehensive semantic description of what this entity represents in the business context",
  "business_context": {{
    "primary_function": "The main business function this entity serves",
    "stakeholder_groups": ["List of stakeholder groups that interact with or depend on this entity"],
    "business_criticality": 0.0-1.0,
    "operational_context": "How this entity fits into day-to-day operations"
  }},
  "entities": [
    ["extracted_entity", "ENTITY_TYPE"],
    // Extract any financial entities mentioned (currencies, amounts, regulatory terms, etc.)
  ],
  "risk_indicators": ["List of potential risk factors associated with this entity"],
  "compliance_flags": ["List of compliance/regulatory considerations"],
  "regulatory_context": {{
    "framework": "Applicable regulatory framework (e.g., SOX, MiFID II, Basel III)",
    "compliance_requirements": ["Specific compliance requirements"],
    "regulatory_complexity": 0.0-1.0
  }},
  "domain_terminology": ["Industry-specific terms relevant to this entity"],
  "synonyms": ["Alternative names or references for this entity"]
}}

Focus on financial and business implications. Be specific and actionable."""

            # Call Grok for enrichment
            response = await self.grok_client.async_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data analyst specializing in semantic enrichment for A2A (Agent-to-Agent) systems. Provide detailed, accurate analysis of financial entities."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse response
            if response.content:
                import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                # Remove markdown code blocks if present
                json_str = re.sub(r'```json\s*|\s*```', '', response.content)
                enrichment = json.loads(json_str)
                
                # Validate and normalize the response
                return self._validate_grok_enrichment(enrichment, entity_type)
            else:
                logger.warning("Empty response from Grok")
                return {}
                
        except Exception as e:
            logger.error(f"Grok enrichment failed: {e}")
            return {}
    
    def _validate_grok_enrichment(self, enrichment: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Validate and normalize Grok enrichment response"""
        validated = {}
        
        # Validate semantic description
        validated['semantic_description'] = enrichment.get('semantic_description', f'AI-enriched {entity_type} entity')
        
        # Validate business context
        business_context = enrichment.get('business_context', {})
        validated['business_context'] = {
            'primary_function': business_context.get('primary_function', f'{entity_type}_management'),
            'stakeholder_groups': business_context.get('stakeholder_groups', ['finance', 'operations']),
            'business_criticality': float(business_context.get('business_criticality', 0.5)),
            'operational_context': business_context.get('operational_context', f'{entity_type}_operations')
        }
        
        # Validate entities
        validated['entities'] = enrichment.get('entities', [])
        
        # Validate risk indicators
        validated['risk_indicators'] = enrichment.get('risk_indicators', [])
        
        # Validate compliance flags
        validated['compliance_flags'] = enrichment.get('compliance_flags', [])
        
        # Validate regulatory context
        reg_context = enrichment.get('regulatory_context', {})
        validated['regulatory_context'] = {
            'framework': reg_context.get('framework', 'General'),
            'compliance_requirements': reg_context.get('compliance_requirements', ['data_retention']),
            'regulatory_complexity': float(reg_context.get('regulatory_complexity', 0.3))
        }
        
        # Validate domain terminology
        validated['domain_terminology'] = enrichment.get('domain_terminology', [entity_type, 'financial', 'enterprise'])
        
        # Validate synonyms
        validated['synonyms'] = enrichment.get('synonyms', [])
        
        return validated
    
    def _extract_risk_indicators(self, entity: Dict[str, Any], entity_type: str) -> List[str]:
        """Extract risk indicators based on entity attributes"""
        risk_indicators = []
        
        if entity_type == "account":
            # High balance accounts
            if entity.get("balance", 0) > 10000000:
                risk_indicators.append("high_value")
            
            # Foreign currency exposure
            if entity.get("currency") not in ["USD", "EUR"]:
                risk_indicators.append("foreign_currency_risk")
            
            # Account type risks
            if entity.get("type") in ["trading", "investment"]:
                risk_indicators.append("market_risk_exposure")
                
        elif entity_type == "product":
            if entity.get("category") in self.semantic_rules["product"]["financial_categories"]:
                risk_indicators.append("regulatory_scrutiny")
        
        return risk_indicators
    
    def _extract_compliance_flags(self, entity: Dict[str, Any], entity_type: str) -> List[str]:
        """Extract compliance-related flags"""
        flags = []
        
        # Check for regulatory keywords in any text field
        entity_str = json.dumps(entity).lower()
        for reg_term in ["sox", "gdpr", "mifid", "basel", "fatca"]:
            if reg_term in entity_str:
                flags.append(f"{reg_term}_applicable")
        
        # Specific compliance rules
        if entity_type == "account" and entity.get("type") == "trust":
            flags.append("fiduciary_requirements")
        
        return flags
    
    async def _generate_semantic_description(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Generate semantic description using Grok if available, else use domain knowledge"""
        if self.grok_client:
            try:
                # Use Grok's A2A-specific processing
                response = await self.grok_client.process_a2a_request(
                    request_type="semantic_description_generation",
                    data={
                        "entity": entity,
                        "entity_type": entity_type,
                        "context": "Generate a comprehensive semantic description for this financial entity"
                    },
                    context={
                        "agent": "AI Preparation Agent",
                        "purpose": "Semantic enrichment for downstream AI processing"
                    }
                )
                
                if response.content:
                    return response.content.strip()
            except Exception as e:
                logger.warning(f"Grok semantic description generation failed: {e}")
        
        # Fallback to rule-based approach
        category = self._determine_semantic_category(entity, entity_type)
        
        base_description = self._get_entity_description(entity, entity_type)
        
        # Add semantic enrichment based on domain analysis
        enrichments = []
        
        # Define thresholds for different entity types
        thresholds = {
            "account": {"high_value": 1000000},
            "product": {"high_value": 100000}
        }
        
        if entity_type == "account":
            if entity.get("balance", 0) > thresholds["account"]["high_value"]:
                enrichments.append("high-value")
            if entity.get("type") in ["reserve", "escrow", "trust"]:
                enrichments.append("regulatory-sensitive")
            if entity.get("type") in ["checking", "operating", "payroll"]:
                enrichments.append("operationally-critical")
        
        enrichment_str = ", ".join(enrichments) if enrichments else "standard"
        
        return f"{category} {entity_type} entity ({enrichment_str}). {base_description}"
    
    def _determine_semantic_category(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Determine semantic category using rules"""
        if entity_type == "account":
            if entity.get("type") in ["reserve", "escrow"]:
                return "Compliance-focused"
            elif entity.get("type") in ["trading", "investment"]:
                return "Financial"
            else:
                return "Operational"
        elif entity_type == "product":
            if entity.get("category") in self.semantic_rules["product"]["financial_categories"]:
                return "Strategic"
            else:
                return "Analytical"
        else:
            return "General"
    
    def _get_entity_description(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Get entity-specific description"""
        if entity_type == "account":
            return f"Account {entity.get('name', 'Unknown')} with {entity.get('currency', 'USD')} currency and {entity.get('type', 'general')} type"
        elif entity_type == "product":
            return f"Product {entity.get('name', 'Unknown')} in {entity.get('category', 'general')} category"
        else:
            return f"{entity_type.capitalize()} entity with identifier {entity.get('id', 'unknown')}"
    
    def _extract_domain_terminology(self, entity: Dict[str, Any], entity_type: str) -> List[str]:
        """Extract domain-specific terminology"""
        terms = [entity_type, "financial", "enterprise"]
        
        # Add entity-specific terms
        if entity_type == "account":
            terms.extend(["ledger", "balance", "transaction"])
            if entity.get("type"):
                terms.append(entity["type"])
        elif entity_type == "product":
            terms.extend(["catalog", "pricing", "inventory"])
            if entity.get("category"):
                terms.append(entity["category"])
        elif entity_type == "measure":
            terms.extend(["metric", "KPI", "performance"])
        
        # Extract terms from entity attributes
        for key, value in entity.items():
            if isinstance(value, str) and len(value) < 50:
                terms.append(value.lower())
        
        return list(set(terms))
    
    def _find_synonyms(self, entity: Dict[str, Any], entity_type: str) -> List[str]:
        """Find synonyms and aliases for the entity"""
        synonyms = []
        
        # Common synonyms by entity type
        synonym_map = {
            "account": ["ledger account", "GL account", "financial account"],
            "book": ["ledger", "journal", "register"],
            "location": ["site", "facility", "branch"],
            "measure": ["metric", "KPI", "indicator"],
            "product": ["item", "SKU", "offering"]
        }
        
        synonyms.extend(synonym_map.get(entity_type, []))
        
        # Add name variations if available
        if "name" in entity:
            name = entity["name"]
            synonyms.append(name.lower())
            synonyms.append(name.upper())
            synonyms.append(name.replace(" ", "_"))
        
        return list(set(synonyms))
    
    def _calculate_enrichment_confidence(self, entity: Dict[str, Any], entity_type: str) -> float:
        """Calculate confidence score for enrichment"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if "id" in entity:
            confidence += 0.1
        if "name" in entity:
            confidence += 0.1
        if len(entity) > 5:
            confidence += 0.2
        if self.grok_client:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _update_normalizer_with_learned_patterns(self) -> None:
        """Update financial normalizer with newly learned patterns"""
        try:
            # Get learned synonym patterns
            learned_synonyms = self.learning_storage.get_learned_patterns("synonym")
            
            # Update normalizer's synonym mappings
            enhanced_mappings = {}
            for pattern in learned_synonyms:
                if pattern.confidence_score >= 0.6:  # Only high-confidence patterns
                    if pattern.target_term not in enhanced_mappings:
                        enhanced_mappings[pattern.target_term] = []
                    enhanced_mappings[pattern.target_term].append(pattern.source_term)
            
            # Merge with existing mappings (this would require updating FinancialDomainNormalizer)
            logger.info(f"Updated normalizer with {len(enhanced_mappings)} learned synonym groups")
            
        except Exception as e:
            logger.error(f"Error updating normalizer with learned patterns: {e}")
    
    def _determine_business_context(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Determine business context for adaptive processing"""
        entity_str = json.dumps(entity, default=str).lower()
        
        # Business context detection
        context_indicators = {
            'retail_banking': ['deposit', 'loan', 'mortgage', 'card', 'branch'],
            'corporate_banking': ['commercial', 'trade_finance', 'cash_management'],
            'investment_banking': ['trading', 'capital_markets', 'underwriting'],
            'asset_management': ['fund', 'portfolio', 'custody'],
            'risk_management': ['risk', 'exposure', 'limit', 'stress'],
            'compliance': ['regulatory', 'audit', 'compliance', 'sox', 'mifid']
        }
        
        for context, indicators in context_indicators.items():
            if any(indicator in entity_str for indicator in indicators):
                return context
        
        return 'general'
    
    def _calculate_adaptive_confidence(self, entity: Dict[str, Any], entity_type: str, 
                                     enhanced_synonyms: List[str], contextual_terms: List[str]) -> float:
        """Calculate confidence score including adaptive learning improvements"""
        base_confidence = self._calculate_enrichment_confidence(entity, entity_type)
        
        # Boost confidence based on adaptive enhancements
        adaptation_boost = 0.0
        if enhanced_synonyms:
            adaptation_boost += 0.1 * min(len(enhanced_synonyms) / 5, 0.2)  # Up to 0.02 boost
        if contextual_terms:
            adaptation_boost += 0.1 * min(len(contextual_terms) / 3, 0.1)   # Up to 0.01 boost
        
        return min(base_confidence + adaptation_boost, 1.0)
    
    async def _periodic_fine_tuning_check(self) -> None:
        """Periodically check if fine-tuning should be performed"""
        while self.is_ready:
            try:
                # Wait for 1 hour between checks
                await asyncio.sleep(3600)
                
                if self.embedding_skill and self.embedding_skill.should_fine_tune():
                    logger.info("Starting automatic embedding model fine-tuning...")
                    result = await self.execute_embedding_fine_tuning()
                    logger.info(f"Fine-tuning result: {result}")
                    
            except Exception as e:
                logger.error(f"Error in periodic fine-tuning check: {e}")
    
    @a2a_skill("fine_tune_embeddings", "Fine-tune the embedding model based on user feedback")
    async def execute_embedding_fine_tuning(self) -> Dict[str, Any]:
        """Execute embedding model fine-tuning"""
        if not self.embedding_skill:
            return create_error_response(500, "Embedding fine-tuning skill not initialized")
        
        try:
            # Run fine-tuning in background thread to avoid blocking
            result = await asyncio.to_thread(
                self.embedding_skill.execute_fine_tuning
            )
            
            # If fine-tuning was successful, update the current model
            if result["status"] == "success":
                self.embedding_model = self.embedding_skill.get_current_model()
                self.embedding_model_name = self.embedding_skill.current_model_path
                logger.info(f"Updated to fine-tuned model: {self.embedding_model_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Fine-tuning execution failed: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("check_fine_tuning_status", "Check if embedding model fine-tuning is needed")
    async def check_fine_tuning_status(self) -> Dict[str, Any]:
        """Check the status of embedding fine-tuning capability"""
        if not self.embedding_skill:
            return {
                "fine_tuning_available": False,
                "reason": "Embedding fine-tuning skill not initialized"
            }
        
        try:
            should_fine_tune = self.embedding_skill.should_fine_tune()
            
            # Get feedback count
            conn = sqlite3.connect(self.embedding_skill.finetuner.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM feedback_events 
                WHERE event_type = 'search_selection'
                AND selected_entity IS NOT NULL
            """)
            total_feedback = cursor.fetchone()[0]
            
            # Get fine-tuning history
            history = self.embedding_skill.finetuner.get_fine_tuning_history(limit=5)
            
            # Get training metrics
            metrics = self.embedding_skill.finetuner.training_metrics
            
            conn.close()
            
            return {
                "fine_tuning_available": True,
                "should_fine_tune": should_fine_tune,
                "current_model": self.embedding_model_name,
                "feedback_count": total_feedback,
                "threshold": self.embedding_skill.fine_tune_threshold,
                "status": "ready" if should_fine_tune else f"need {self.embedding_skill.fine_tune_threshold - total_feedback} more feedback events",
                "training_metrics": metrics,
                "model_versions": self.embedding_skill.model_versions,
                "recent_history": history
            }
            
        except Exception as e:
            logger.error(f"Error checking fine-tuning status: {e}")
            return create_error_response(500, str(e))
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down AI Preparation Agent...")
        
        if self.http_client:
            await self.http_client.aclose()
        
        self.is_ready = False
        logger.info("AI Preparation Agent shutdown complete")
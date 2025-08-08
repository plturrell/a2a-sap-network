"""
AI Data Readiness & Vectorization Agent
Transforms standardized financial entities into AI-ready semantic objects with vector embeddings
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import logging
import json

from ..core.a2a_types import A2AMessage, MessagePart, MessageRole
from ..core.message_queue import MessageQueue
from .data_standardization_agent import TaskState, TaskStatus, TaskArtifact, AgentCard
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor

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
    confidence: float
    evidence: List[str]


class VectorEmbeddings(BaseModel):
    semantic: List[float]
    hierarchical: List[float]
    contextual: List[float]
    relationship: List[float]
    quality: List[float]
    temporal: List[float]
    composite: List[float]


class AIReadyEntity(BaseModel):
    entity_id: str
    entity_type: str
    original_data: Dict[str, Any]
    semantic_enrichment: SemanticEnrichment
    relationships: List[EntityRelationship]
    embeddings: VectorEmbeddings
    ai_readiness_metadata: Dict[str, Any]


class AIReadinessValidationResult(BaseModel):
    ready_for_ai: bool
    overall_readiness_score: float
    embedding_quality: float
    relationship_completeness: float
    semantic_coherence: float
    issues: List[str]
    validation_timestamp: str


class AIPreparationAgent(AgentHelpSeeker):
    """AI Data Readiness & Vectorization Agent for A2A Protocol"""
    
    def __init__(self, base_url: str, agent_id: str = "ai_preparation_agent_2"):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        self.agent_id = agent_id
        self.agent_name = "AI Data Readiness & Vectorization Agent"
        self.version = "1.0.0"
        self.protocol_version = "0.2.9"
        
        # Initialize message queue with agent-specific configuration (aligned with other agents)
        self.initialize_message_queue(
            agent_id=self.agent_id,
            max_concurrent_processing=3,  # Moderate for AI processing
            auto_mode_threshold=6,        # Switch to queue after 6 pending
            enable_streaming=True,        # Support real-time processing
            enable_batch_processing=True  # Support batch processing
        )
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name="AI Preparation Agent"
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "agent_type": "ai_preparation",
            "timeout": 30.0,
            "retry.max_attempts": 3
        }
        self.initialize_help_action_system(self.agent_id, agent_context)
        
        # Initialize Database-backed AI Decision Logger
        from ..core.ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
        
        # Construct Data Manager URL
        data_manager_url = f"{self.base_url.replace('/agents/', '/').rstrip('/')}/data-manager"
        
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=self.agent_id,
            data_manager_url=data_manager_url,
            memory_size=600,  # Moderate memory for AI preparation decisions
            learning_threshold=5,  # Lower threshold for AI-focused learning
            cache_ttl=300
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI advisor for enhanced capabilities
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_type="ai_preparation",
            specialized_tasks=["semantic_enrichment", "vector_embeddings", "knowledge_graph"]
        )
        
        # Initialize ORD lineage tracking
        self._current_ord_lineage = {}
        
        # Agent capabilities (aligned with other agents)
        self.capabilities = {
            "streaming": True,
            "pushNotifications": True,
            "stateTransitionHistory": True,
            "batchProcessing": True,
            "smartContractDelegation": True,
            "aiAdvisor": True,
            "helpSeeking": True,
            "taskTracking": True
        }
        
        # Skills definition
        self.skills = [
            {
                "id": "semantic-context-enrichment",
                "name": "Semantic Context Enrichment",
                "description": "Add rich semantic context, business descriptions, and domain-specific terminology",
                "tags": ["semantic", "context", "nlp", "business-intelligence"]
            },
            {
                "id": "entity-relationship-discovery",
                "name": "Entity Relationship Discovery", 
                "description": "Discover and map relationships between entities across different financial dimensions",
                "tags": ["relationships", "graph", "discovery", "cross-entity"]
            },
            {
                "id": "multi-dimensional-feature-extraction",
                "name": "Multi-Dimensional Feature Extraction",
                "description": "Extract semantic, hierarchical, contextual, and quality features for vector embeddings",
                "tags": ["features", "vectorization", "multi-dimensional", "ai-ready"]
            },
            {
                "id": "vector-embedding-generation", 
                "name": "Vector Embedding Generation",
                "description": "Generate specialized vector embeddings optimized for financial domain understanding",
                "tags": ["embeddings", "vectors", "neural", "domain-specific"]
            },
            {
                "id": "knowledge-graph-structuring",
                "name": "Knowledge Graph Structuring",
                "description": "Structure entities and relationships for RDF knowledge graph representation",
                "tags": ["rdf", "knowledge-graph", "ontology", "turtle"]
            },
            {
                "id": "ai-readiness-validation",
                "name": "AI Readiness Validation",
                "description": "Validate data quality and completeness for AI processing and knowledge graph ingestion",
                "tags": ["validation", "quality", "ai-readiness", "completeness"]
            }
        ]
        
        logger.info(f"Initialized {self.agent_name} with ID: {self.agent_id}")

    def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card for A2A discovery"""
        return {
            "name": self.agent_name,
            "description": "Transforms standardized financial entities into AI-ready semantic objects with multi-dimensional vector embeddings",
            "url": self.base_url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "provider": {
                "organization": "Financial AI Processing Services",
                "url": "https://financial-ai.example.com"
            },
            "capabilities": self.capabilities,
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json", "text/turtle", "application/x-ndjson"],
            "skills": self.skills,
            "metadata": {
                "tags": ["financial", "ai", "vectorization", "knowledge-graph", "semantic"],
                "categories": ["data-processing", "ai-preparation", "semantic-enrichment"],
                "domain": "financial-services",
                "compliance": ["data-privacy", "financial-regulations"],
                "integration": {
                    "upstreamAgents": ["financial-data-standardization-agent"],
                    "downstreamSystems": ["sap-knowledge-engine", "vector-databases"],
                    "supportedFormats": ["json", "turtle", "ndjson"]
                }
            }
        }

    async def process_message(self, message: A2AMessage, context_id: str, priority: str = "medium") -> Dict[str, Any]:
        """Process incoming A2A messages for AI preparation"""
        task_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing AI preparation message - Task: {task_id}, Context: {context_id}")
            
            # Initialize task tracking with proper TaskTracker
            task_priority = TaskPriority.HIGH if priority == "high" else TaskPriority.MEDIUM if priority == "medium" else TaskPriority.LOW
            
            await self.task_tracker.create_task(
                task_id=task_id,
                description="AI preparation processing",
                priority=task_priority,
                context_id=context_id,
                checklist_items=[]
            )
            
            # Parse standardized entities from input and extract ORD lineage
            standardized_entities, ord_lineage = await self._parse_standardized_entities(message)
            
            if not standardized_entities:
                raise ValueError("No standardized entities found in input message")
            
            logger.info(f"🔗 Processing {len(standardized_entities)} entities with ORD lineage from registration: {ord_lineage.get('original_registration_id', 'unknown')}")
            
            # Update task progress
            await self.task_tracker.update_task_progress(task_id, 10, "Parsing input data")
            
            # Step 1: Semantic Context Enrichment
            enriched_entities = await self._enrich_semantic_context(standardized_entities)
            await self.task_tracker.update_task_progress(task_id, 25, "Semantic enrichment completed")
            
            # Step 2: Entity Relationship Discovery
            relationship_mapped_entities = await self._discover_relationships(enriched_entities)
            await self.task_tracker.update_task_progress(task_id, 40, "Relationship discovery completed")
            
            # Step 3: Multi-Dimensional Feature Extraction
            feature_sets = await self._extract_features(relationship_mapped_entities)
            await self.task_tracker.update_task_progress(task_id, 55, "Feature extraction completed")
            
            # Step 4: Vector Embedding Generation
            embedded_entities = await self._generate_embeddings(feature_sets)
            await self.task_tracker.update_task_progress(task_id, 75, "Embedding generation completed")
            
            # Step 5: Knowledge Graph Structuring
            knowledge_graph_data = await self._structure_knowledge_graph(embedded_entities)
            await self.task_tracker.update_task_progress(task_id, 90, "Knowledge graph structuring completed")
            
            # Step 6: AI Readiness Validation
            validation_result = await self._validate_ai_readiness(embedded_entities, knowledge_graph_data)
            
            if not validation_result.ready_for_ai:
                await self.task_tracker.mark_task_failed(task_id, f"AI readiness validation failed: {', '.join(validation_result.issues)}")
                raise ValueError(f"AI readiness validation failed: {', '.join(validation_result.issues)}")
            
            # Finalize task
            await self.task_tracker.complete_task(task_id, "AI preparation completed successfully")
            
            # Prepare response with complete ORD lineage preservation
            response_data = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "success",
                "context_id": context_id,
                "processed_entities": len(embedded_entities),
                "ai_ready_entities": embedded_entities,
                "knowledge_graph_rdf": knowledge_graph_data["turtle_format"],
                "vector_index": knowledge_graph_data["vector_index"],
                "validation_report": validation_result.dict(),
                "ord_lineage": {
                    **ord_lineage,
                    "ai_processing_complete": True,
                    "ai_processed_at": datetime.utcnow().isoformat(),
                    "pipeline_stage": "ai_preparation_complete",
                    "data_transformation_chain": [
                        ord_lineage.get("provenance", {}).get("data_transformation", "unknown"),
                        "standardized_to_ai_ready"
                    ]
                },
                "data_provenance": {
                    **ord_lineage.get("provenance", {}),
                    "ai_processed_by": self.agent_id,
                    "ai_processing_timestamp": datetime.utcnow().isoformat(),
                    "complete_pipeline": "data_product_agent → financial_standardization_agent → ai_preparation_agent"
                },
                "ingestion_metadata": {
                    "total_entities": len(embedded_entities),
                    "embedding_dimensions": self._get_embedding_dimensions(embedded_entities),
                    "knowledge_graph_triples": knowledge_graph_data["triple_count"],
                    "readiness_score": validation_result.overall_readiness_score,
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "original_data_source": ord_lineage.get("original_registration_id", "unknown")
                },
                "processing_summary": {
                    "entities_processed": len(embedded_entities),
                    "relationships_discovered": sum(len(e["relationships"]) for e in embedded_entities),
                    "embeddings_generated": len(embedded_entities) * 6,  # 6 embedding types per entity
                    "validation_score": validation_result.overall_readiness_score,
                    "data_lineage_preserved": bool(ord_lineage.get("original_registration_id"))
                }
            }
            
            logger.info(f"AI preparation completed - Task: {task_id}, Entities: {len(embedded_entities)}, Score: {validation_result.overall_readiness_score}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"AI preparation failed - Task: {task_id}, Error: {str(e)}")
            
            # Mark task as failed in task tracker
            await self.task_tracker.mark_task_failed(task_id, str(e))
            
            return {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "error",
                "context_id": context_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _parse_standardized_entities(self, message: A2AMessage) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse standardized entities from incoming message and extract ORD lineage"""
        try:
            for part in message.parts:
                if part.kind == "data" and isinstance(part.data, dict):
                    # Extract ORD lineage information for data traceability
                    ord_lineage = part.data.get("ord_lineage", {})
                    data_provenance = part.data.get("data_provenance", {})
                    
                    # Store lineage information in instance for later use
                    self._current_ord_lineage = {
                        **ord_lineage,
                        "provenance": data_provenance,
                        "received_at": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"📋 Received data with ORD lineage - Registration ID: {ord_lineage.get('original_registration_id', 'unknown')}")
                    
                    # Check for standardized results from previous agent
                    if "standardized_results" in part.data:
                        standardized_results = part.data["standardized_results"]
                        entities = self._flatten_batch_results(standardized_results)
                        return entities, self._current_ord_lineage
                    # Check for direct entity data
                    elif "entities" in part.data:
                        return part.data["entities"], self._current_ord_lineage
                    # Check for nested data structure
                    elif "data" in part.data and isinstance(part.data["data"], dict):
                        nested_data = part.data["data"]
                        if "standardized_results" in nested_data:
                            standardized_results = nested_data["standardized_results"]
                            entities = self._flatten_batch_results(standardized_results)
                            return entities, self._current_ord_lineage
            
            # If no explicit standardized results found, try to extract any entity-like data
            logger.warning("No explicit standardized_results found, attempting to parse raw data")
            return [], {}
            
        except Exception as e:
            logger.error(f"Failed to parse standardized entities: {e}")
            return [], {}

    def _flatten_batch_results(self, results) -> List[Dict[str, Any]]:
        """Convert batch format results to flat entity list"""
        try:
            # If already a flat list, return as-is
            if isinstance(results, list):
                entities = []
                for item in results:
                    if isinstance(item, dict):
                        validated_entity = self._validate_and_enrich_entity(item)
                        if validated_entity:
                            entities.append(validated_entity)
                return entities
            
            # If batch format (dict with entity types as keys)
            elif isinstance(results, dict):
                entities = []
                for entity_type, records in results.items():
                    if isinstance(records, list):
                        for record in records:
                            if isinstance(record, dict):
                                # Ensure entity_type is set
                                record["entity_type"] = entity_type
                                validated_entity = self._validate_and_enrich_entity(record)
                                if validated_entity:
                                    entities.append(validated_entity)
                return entities
            
            logger.warning(f"Unexpected results format: {type(results)}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to flatten batch results: {e}")
            return []

    def _validate_and_enrich_entity(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate entity structure and enrich with missing required fields"""
        try:
            # Required fields for AI preparation
            required_fields = ["entity_id", "entity_type"]
            
            # Ensure entity_id exists
            if not entity.get("entity_id"):
                entity["entity_id"] = str(uuid.uuid4())
            
            # Ensure entity_type exists
            if not entity.get("entity_type"):
                # Try to infer from content
                entity["entity_type"] = self._infer_entity_type(entity)
            
            # Validate required fields are present
            missing_fields = [field for field in required_fields if not entity.get(field)]
            if missing_fields:
                logger.warning(f"Entity missing required fields: {missing_fields}. Skipping entity: {entity.get('entity_id', 'unknown')}")
                return None
            
            # Enrich with default values for optional fields that AI preparation expects
            if not entity.get("clean_name"):
                entity["clean_name"] = entity.get("name", entity.get("original_name", f"Entity_{entity['entity_id'][:8]}"))
            
            if not entity.get("hierarchy_path"):
                entity["hierarchy_path"] = f"/{entity['entity_type']}/{entity.get('clean_name', entity['entity_id'])}"
            
            # Add processing metadata
            entity["ai_prep_validated"] = True
            entity["validated_at"] = datetime.utcnow().isoformat()
            
            return entity
            
        except Exception as e:
            logger.error(f"Failed to validate entity {entity.get('entity_id', 'unknown')}: {e}")
            return None

    def _infer_entity_type(self, entity: Dict[str, Any]) -> str:
        """Infer entity type from entity content"""
        entity_str = str(entity).lower()
        
        # Simple heuristic-based inference
        if any(term in entity_str for term in ["location", "country", "region", "geography"]):
            return "location"
        elif any(term in entity_str for term in ["account", "ledger", "gl"]):
            return "account"
        elif any(term in entity_str for term in ["product", "service", "offering"]):
            return "product"
        elif any(term in entity_str for term in ["book", "portfolio", "trading"]):
            return "book"
        elif any(term in entity_str for term in ["measure", "metric", "kpi"]):
            return "measure"
        else:
            return "unknown"

    async def _enrich_semantic_context(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich entities with semantic context and business intelligence"""
        enriched_entities = []
        
        for entity in entities:
            try:
                # Generate semantic description
                semantic_description = await self._generate_semantic_description(entity)
                
                # Extract business context
                business_context = await self._extract_business_context(entity)
                
                # Extract domain terminology
                domain_terminology = await self._extract_domain_terminology(entity)
                
                # Extract regulatory context
                regulatory_context = await self._extract_regulatory_context(entity)
                
                # Generate synonyms and aliases
                synonyms_aliases = await self._generate_synonyms_aliases(entity)
                
                # Create enrichment object
                enrichment = {
                    "semantic_description": semantic_description,
                    "business_context": business_context,
                    "domain_terminology": domain_terminology,
                    "regulatory_context": regulatory_context,
                    "synonyms_and_aliases": synonyms_aliases,
                    "contextual_metadata": {
                        "enrichment_timestamp": datetime.utcnow().isoformat(),
                        "confidence_score": 0.85,
                        "source_agent": self.agent_id
                    }
                }
                
                # Add enrichment to entity
                enriched_entity = {
                    **entity,
                    "semantic_enrichment": enrichment
                }
                
                enriched_entities.append(enriched_entity)
                
            except Exception as e:
                logger.error(f"Failed to enrich entity {entity.get('entity_id', 'unknown')}: {e}")
                # Add entity without enrichment
                enriched_entities.append(entity)
        
        return enriched_entities

    async def _generate_semantic_description(self, entity: Dict[str, Any]) -> str:
        """Generate rich semantic description for entity"""
        components = []
        
        entity_type = entity.get("entity_type", "unknown")
        clean_name = entity.get("clean_name", entity.get("name", "unnamed"))
        
        components.append(f"{entity_type} entity named '{clean_name}'")
        
        if entity.get("hierarchy_path"):
            components.append(f"positioned in hierarchy: {entity['hierarchy_path']}")
        
        if entity.get("classification"):
            components.append(f"classified as {entity['classification']}")
        
        if entity.get("geographic_context"):
            components.append(f"associated with {entity['geographic_context']}")
        
        if entity.get("business_line"):
            components.append(f"part of {entity['business_line']} business line")
        
        if entity.get("regulatory_framework"):
            components.append(f"governed by {entity['regulatory_framework']} framework")
        
        return ". ".join(components) + "."

    async def _extract_business_context(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business context from entity"""
        return {
            "primary_function": self._infer_primary_function(entity),
            "stakeholder_groups": self._identify_stakeholders(entity),
            "business_criticality": await self._assess_business_criticality(entity),
            "operational_context": self._extract_operational_context(entity),
            "strategic_importance": await self._assess_strategic_importance(entity)
        }

    def _infer_primary_function(self, entity: Dict[str, Any]) -> str:
        """Infer primary business function of entity"""
        entity_type = entity.get("entity_type", "").lower()
        
        function_map = {
            "account": "Financial transaction processing and reporting",
            "location": "Geographic operational management",
            "product": "Business offering and service delivery",
            "measure": "Performance measurement and analytics",
            "book": "Risk and regulatory reporting"
        }
        
        return function_map.get(entity_type, "General business entity")

    def _identify_stakeholders(self, entity: Dict[str, Any]) -> List[str]:
        """Identify stakeholder groups for entity"""
        stakeholders = ["Internal Finance Team"]
        
        entity_type = entity.get("entity_type", "").lower()
        
        if entity_type == "account":
            stakeholders.extend(["Accounting Department", "Auditors", "Regulatory Bodies"])
        elif entity_type == "location":
            stakeholders.extend(["Operations Team", "Compliance Officers", "Local Management"])
        elif entity_type == "product":
            stakeholders.extend(["Product Managers", "Sales Team", "Customers"])
        elif entity_type == "measure":
            stakeholders.extend(["Business Analysts", "Executive Management", "Risk Managers"])
        elif entity_type == "book":
            stakeholders.extend(["Risk Management", "Regulatory Reporting", "Senior Management"])
        
        return stakeholders

    async def _assess_business_criticality(self, entity: Dict[str, Any]) -> float:
        """Assess business criticality score (0.0 to 1.0)"""
        # Simple heuristic based on entity type and characteristics
        base_criticality = {
            "account": 0.8,
            "location": 0.6,
            "product": 0.7,
            "measure": 0.9,
            "book": 0.95
        }
        
        entity_type = entity.get("entity_type", "").lower()
        criticality = base_criticality.get(entity_type, 0.5)
        
        # Adjust based on regulatory framework
        if entity.get("regulatory_framework"):
            criticality = min(1.0, criticality + 0.1)
        
        return criticality

    def _extract_operational_context(self, entity: Dict[str, Any]) -> str:
        """Extract operational context description"""
        contexts = []
        
        if entity.get("geographic_context"):
            contexts.append(f"operates in {entity['geographic_context']}")
        
        if entity.get("business_line"):
            contexts.append(f"supports {entity['business_line']} operations")
        
        if entity.get("regulatory_framework"):
            contexts.append(f"complies with {entity['regulatory_framework']} regulations")
        
        return "; ".join(contexts) if contexts else "general operational context"

    async def _assess_strategic_importance(self, entity: Dict[str, Any]) -> float:
        """Assess strategic importance score (0.0 to 1.0)"""
        # Simple scoring based on entity characteristics
        importance = 0.5
        
        if entity.get("hierarchy_path") and len(entity["hierarchy_path"].split("/")) <= 2:
            importance += 0.2  # High-level entities are more strategic
        
        if entity.get("regulatory_framework"):
            importance += 0.15  # Regulatory entities are strategic
        
        if entity.get("business_line"):
            importance += 0.1  # Business line entities are important
        
        return min(1.0, importance)

    async def _extract_domain_terminology(self, entity: Dict[str, Any]) -> List[str]:
        """Extract domain-specific terminology"""
        terminology = []
        
        entity_type = entity.get("entity_type", "").lower()
        
        # Base terminology by type
        type_terms = {
            "account": ["ledger", "general ledger", "chart of accounts", "financial statement"],
            "location": ["jurisdiction", "regulatory region", "operational territory"],
            "product": ["financial instrument", "business offering", "service line"],
            "measure": ["KPI", "metric", "performance indicator", "business measure"],
            "book": ["trading book", "banking book", "risk position", "regulatory capital"]
        }
        
        terminology.extend(type_terms.get(entity_type, []))
        
        # Add entity-specific terms
        if entity.get("classification"):
            terminology.append(entity["classification"])
        
        if entity.get("regulatory_framework"):
            terminology.extend([entity["regulatory_framework"], "compliance", "regulatory reporting"])
        
        return list(set(terminology))  # Remove duplicates

    async def _extract_regulatory_context(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regulatory context information"""
        framework = entity.get("regulatory_framework", "General Financial Regulations")
        
        # Map frameworks to compliance requirements
        compliance_map = {
            "IFRS": ["International Financial Reporting", "Fair Value Measurement", "Financial Instruments"],
            "US GAAP": ["Generally Accepted Accounting Principles", "SEC Reporting", "Sarbanes-Oxley"],
            "Basel III": ["Capital Adequacy", "Liquidity Coverage", "Risk Management"],
            "MiFID": ["Investment Services", "Market Transparency", "Investor Protection"]
        }
        
        requirements = compliance_map.get(framework, ["Standard Financial Compliance"])
        
        return {
            "framework": framework,
            "compliance_requirements": requirements,
            "regulatory_complexity": 0.7 if framework != "General Financial Regulations" else 0.3
        }

    async def _generate_synonyms_aliases(self, entity: Dict[str, Any]) -> List[str]:
        """Generate synonyms and aliases for entity"""
        synonyms = []
        
        clean_name = entity.get("clean_name", "")
        original_name = entity.get("original_name", "")
        
        if clean_name and clean_name != original_name:
            synonyms.append(original_name)
        
        # Add common synonyms based on entity type
        entity_type = entity.get("entity_type", "").lower()
        
        if entity_type == "account":
            synonyms.extend(["GL Account", "Ledger Account"])
        elif entity_type == "location":
            synonyms.extend(["Geography", "Region", "Territory"])
        elif entity_type == "product":
            synonyms.extend(["Service", "Offering", "Line of Business"])
        elif entity_type == "measure":
            synonyms.extend(["Metric", "KPI", "Indicator"])
        elif entity_type == "book":
            synonyms.extend(["Portfolio", "Position", "Risk Book"])
        
        return list(set(synonyms))  # Remove duplicates

    async def _discover_relationships(self, enriched_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover relationships between entities"""
        relationship_mapped_entities = []
        
        for entity in enriched_entities:
            relationships = await self._find_entity_relationships(entity, enriched_entities)
            
            entity_with_relationships = {
                **entity,
                "relationships": relationships,
                "semantic_similarities": await self._calculate_semantic_similarities(entity, enriched_entities),
                "cross_type_connections": await self._find_cross_type_connections(entity, enriched_entities)
            }
            
            relationship_mapped_entities.append(entity_with_relationships)
        
        return relationship_mapped_entities

    async def _find_entity_relationships(self, entity: Dict[str, Any], all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find relationships for a specific entity"""
        relationships = []
        
        entity_hierarchy = entity.get("hierarchy_path", "")
        entity_type = entity.get("entity_type", "")
        
        for other_entity in all_entities:
            if other_entity.get("entity_id") == entity.get("entity_id"):
                continue
            
            other_hierarchy = other_entity.get("hierarchy_path", "")
            other_type = other_entity.get("entity_type", "")
            
            # Check for hierarchical relationships
            if entity_hierarchy and other_hierarchy:
                if other_hierarchy.startswith(entity_hierarchy + "/"):
                    relationships.append({
                        "source_entity": entity.get("entity_id"),
                        "target_entity": other_entity.get("entity_id"),
                        "relationship_type": "parent_child",
                        "confidence": 0.9,
                        "evidence": ["hierarchical_path_analysis"]
                    })
                elif entity_hierarchy.startswith(other_hierarchy + "/"):
                    relationships.append({
                        "source_entity": entity.get("entity_id"),
                        "target_entity": other_entity.get("entity_id"),
                        "relationship_type": "child_parent",
                        "confidence": 0.9,
                        "evidence": ["hierarchical_path_analysis"]
                    })
            
            # Check for type-based relationships
            if entity_type != other_type:
                similarity_score = await self._calculate_cross_type_similarity(entity, other_entity)
                if similarity_score > 0.7:
                    relationships.append({
                        "source_entity": entity.get("entity_id"),
                        "target_entity": other_entity.get("entity_id"),
                        "relationship_type": "cross_type_association",
                        "confidence": similarity_score,
                        "evidence": ["semantic_similarity", "business_context"]
                    })
        
        return relationships

    async def _calculate_semantic_similarities(self, entity: Dict[str, Any], all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate semantic similarities with other entities"""
        similarities = []
        
        entity_description = entity.get("semantic_enrichment", {}).get("semantic_description", "")
        
        for other_entity in all_entities:
            if other_entity.get("entity_id") == entity.get("entity_id"):
                continue
            
            other_description = other_entity.get("semantic_enrichment", {}).get("semantic_description", "")
            
            # Simple text similarity (in real implementation, use semantic embeddings)
            similarity_score = await self._calculate_text_similarity(entity_description, other_description)
            
            if similarity_score > 0.5:
                similarities.append({
                    "target_entity": other_entity.get("entity_id"),
                    "similarity_score": similarity_score,
                    "similarity_type": "semantic_description"
                })
        
        return similarities

    async def _find_cross_type_connections(self, entity: Dict[str, Any], all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find connections across different entity types"""
        connections = []
        
        entity_type = entity.get("entity_type", "")
        entity_geo = entity.get("geographic_context", "")
        entity_business = entity.get("business_line", "")
        
        for other_entity in all_entities:
            if other_entity.get("entity_id") == entity.get("entity_id"):
                continue
            
            other_type = other_entity.get("entity_type", "")
            other_geo = other_entity.get("geographic_context", "")
            other_business = other_entity.get("business_line", "")
            
            # Skip same type
            if entity_type == other_type:
                continue
            
            connection_strength = 0.0
            connection_reasons = []
            
            # Geographic connection
            if entity_geo and other_geo and entity_geo == other_geo:
                connection_strength += 0.3
                connection_reasons.append("geographic_alignment")
            
            # Business line connection
            if entity_business and other_business and entity_business == other_business:
                connection_strength += 0.4
                connection_reasons.append("business_line_alignment")
            
            # Regulatory framework connection
            entity_reg = entity.get("regulatory_framework")
            other_reg = other_entity.get("regulatory_framework")
            if entity_reg and other_reg and entity_reg == other_reg:
                connection_strength += 0.3
                connection_reasons.append("regulatory_alignment")
            
            if connection_strength > 0.5:
                connections.append({
                    "target_entity": other_entity.get("entity_id"),
                    "target_type": other_type,
                    "connection_strength": connection_strength,
                    "connection_reasons": connection_reasons
                })
        
        return connections

    async def _calculate_cross_type_similarity(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calculate similarity between entities of different types"""
        similarity = 0.0
        
        # Geographic similarity
        if entity1.get("geographic_context") == entity2.get("geographic_context"):
            similarity += 0.3
        
        # Business line similarity
        if entity1.get("business_line") == entity2.get("business_line"):
            similarity += 0.4
        
        # Regulatory framework similarity
        if entity1.get("regulatory_framework") == entity2.get("regulatory_framework"):
            similarity += 0.3
        
        return min(1.0, similarity)

    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (placeholder for semantic similarity)"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    async def _extract_features(self, relationship_mapped_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract multi-dimensional features for embedding generation"""
        feature_sets = []
        
        for entity in relationship_mapped_entities:
            features = {
                "semantic_features": await self._extract_semantic_features(entity),
                "hierarchical_features": await self._extract_hierarchical_features(entity),
                "contextual_features": await self._extract_contextual_features(entity),
                "relationship_features": await self._extract_relationship_features(entity),
                "quality_features": await self._extract_quality_features(entity),
                "temporal_features": await self._extract_temporal_features(entity)
            }
            
            feature_set = {
                "entity_id": entity.get("entity_id"),
                "features": features,
                "feature_metadata": {
                    "extraction_timestamp": datetime.utcnow().isoformat(),
                    "feature_completeness": self._calculate_feature_completeness(features),
                    "extraction_confidence": 0.85
                }
            }
            
            feature_sets.append(feature_set)
        
        return feature_sets

    async def _extract_semantic_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic features for embedding"""
        enrichment = entity.get("semantic_enrichment", {})
        
        return {
            "primary_text": enrichment.get("semantic_description", ""),
            "contextual_texts": [
                enrichment.get("business_context", {}).get("primary_function", ""),
                enrichment.get("regulatory_context", {}).get("framework", ""),
                entity.get("clean_name", ""),
                entity.get("hierarchy_path", "")
            ],
            "domain_terms": enrichment.get("domain_terminology", []),
            "synonyms": enrichment.get("synonyms_and_aliases", [])
        }

    async def _extract_hierarchical_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hierarchical features"""
        hierarchy_path = entity.get("hierarchy_path", "")
        hierarchy_level = len(hierarchy_path.split("/")) if hierarchy_path else 0
        
        relationships = entity.get("relationships", [])
        
        return {
            "entity_type": entity.get("entity_type", ""),
            "entity_subtype": entity.get("entity_subtype", ""),
            "hierarchy_level": hierarchy_level,
            "parent_entities": [r["target_entity"] for r in relationships if r["relationship_type"] == "child_parent"],
            "child_entities": [r["target_entity"] for r in relationships if r["relationship_type"] == "parent_child"],
            "sibling_entities": [r["target_entity"] for r in relationships if r["relationship_type"] == "sibling"]
        }

    async def _extract_contextual_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features"""
        enrichment = entity.get("semantic_enrichment", {})
        business_context = enrichment.get("business_context", {})
        
        return {
            "business_criticality": business_context.get("business_criticality", 0.5),
            "geographic_context": entity.get("geographic_context", ""),
            "regulatory_complexity": enrichment.get("regulatory_context", {}).get("regulatory_complexity", 0.3),
            "stakeholder_impact": len(business_context.get("stakeholder_groups", [])),
            "operational_scope": self._assess_operational_scope(entity)
        }

    def _assess_operational_scope(self, entity: Dict[str, Any]) -> float:
        """Assess operational scope (0.0 to 1.0)"""
        scope = 0.5
        
        if entity.get("geographic_context"):
            scope += 0.2
        
        if entity.get("business_line"):
            scope += 0.2
        
        if entity.get("regulatory_framework"):
            scope += 0.1
        
        return min(1.0, scope)

    async def _extract_relationship_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relationship features"""
        relationships = entity.get("relationships", [])
        similarities = entity.get("semantic_similarities", [])
        connections = entity.get("cross_type_connections", [])
        
        return {
            "relationship_count": len(relationships),
            "high_confidence_relationships": len([r for r in relationships if r["confidence"] > 0.8]),
            "cross_type_connections": len(connections),
            "semantic_similarity_score": sum(s["similarity_score"] for s in similarities) / len(similarities) if similarities else 0.0,
            "relationship_diversity": len(set(r["relationship_type"] for r in relationships))
        }

    async def _extract_quality_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality features"""
        return {
            "data_completeness": self._calculate_data_completeness(entity),
            "enrichment_quality": self._calculate_enrichment_quality(entity),
            "relationship_quality": self._calculate_relationship_quality(entity),
            "semantic_coherence": self._calculate_semantic_coherence(entity)
        }

    def _calculate_data_completeness(self, entity: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        required_fields = ["entity_id", "entity_type", "clean_name"]
        optional_fields = ["hierarchy_path", "classification", "geographic_context", "business_line"]
        
        required_score = sum(1 for field in required_fields if entity.get(field)) / len(required_fields)
        optional_score = sum(1 for field in optional_fields if entity.get(field)) / len(optional_fields)
        
        return (required_score * 0.7) + (optional_score * 0.3)

    def _calculate_enrichment_quality(self, entity: Dict[str, Any]) -> float:
        """Calculate enrichment quality score"""
        enrichment = entity.get("semantic_enrichment", {})
        
        if not enrichment:
            return 0.0
        
        quality_indicators = [
            bool(enrichment.get("semantic_description")),
            bool(enrichment.get("business_context")),
            bool(enrichment.get("domain_terminology")),
            bool(enrichment.get("regulatory_context")),
            bool(enrichment.get("synonyms_and_aliases"))
        ]
        
        return sum(quality_indicators) / len(quality_indicators)

    def _calculate_relationship_quality(self, entity: Dict[str, Any]) -> float:
        """Calculate relationship quality score"""
        relationships = entity.get("relationships", [])
        
        if not relationships:
            return 0.0
        
        high_confidence_count = sum(1 for r in relationships if r["confidence"] > 0.7)
        return high_confidence_count / len(relationships)

    def _calculate_semantic_coherence(self, entity: Dict[str, Any]) -> float:
        """Calculate semantic coherence score"""
        # Simple coherence check based on consistency of data
        coherence_score = 0.8  # Default score
        
        # Check consistency between entity type and classification
        entity_type = entity.get("entity_type", "").lower()
        classification = entity.get("classification", "").lower()
        
        if entity_type in classification or classification in entity_type:
            coherence_score += 0.1
        
        # Check consistency between hierarchy and entity type
        hierarchy = entity.get("hierarchy_path", "").lower()
        if entity_type in hierarchy:
            coherence_score += 0.1
        
        return min(1.0, coherence_score)

    async def _extract_temporal_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features"""
        current_time = datetime.utcnow()
        
        return {
            "creation_timestamp": entity.get("created_at", current_time.isoformat()),
            "last_modified": entity.get("updated_at", current_time.isoformat()),
            "processing_timestamp": current_time.isoformat(),
            "temporal_relevance": 1.0,  # Assume current data is fully relevant
            "data_freshness": 1.0  # Assume data is fresh
        }

    def _calculate_feature_completeness(self, features: Dict[str, Any]) -> float:
        """Calculate feature completeness score"""
        feature_types = ["semantic_features", "hierarchical_features", "contextual_features", 
                        "relationship_features", "quality_features", "temporal_features"]
        
        completeness_scores = []
        
        for feature_type in feature_types:
            feature_data = features.get(feature_type, {})
            if isinstance(feature_data, dict):
                non_empty_fields = sum(1 for v in feature_data.values() if v)
                total_fields = len(feature_data)
                completeness_scores.append(non_empty_fields / total_fields if total_fields > 0 else 0.0)
        
        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

    async def _generate_embeddings(self, feature_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate vector embeddings from features"""
        embedded_entities = []
        
        for feature_set in feature_sets:
            try:
                features = feature_set["features"]
                
                # Generate different types of embeddings
                embeddings = {
                    "semantic": await self._generate_semantic_embedding(features["semantic_features"]),
                    "hierarchical": await self._generate_hierarchical_embedding(features["hierarchical_features"]),
                    "contextual": await self._generate_contextual_embedding(features["contextual_features"]),
                    "relationship": await self._generate_relationship_embedding(features["relationship_features"]),
                    "quality": await self._generate_quality_embedding(features["quality_features"]),
                    "temporal": await self._generate_temporal_embedding(features["temporal_features"])
                }
                
                # Generate composite embedding
                composite_embedding = await self._generate_composite_embedding(embeddings)
                embeddings["composite"] = composite_embedding
                
                embedded_entity = {
                    "entity_id": feature_set["entity_id"],
                    "embeddings": embeddings,
                    "embedding_metadata": {
                        "dimensions": self._get_single_entity_embedding_dimensions(embeddings),
                        "generation_timestamp": datetime.utcnow().isoformat(),
                        "model_versions": self._get_model_versions(),
                        "quality_score": await self._assess_embedding_quality(embeddings)
                    }
                }
                
                embedded_entities.append(embedded_entity)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for entity {feature_set.get('entity_id', 'unknown')}: {e}")
        
        return embedded_entities

    async def _generate_semantic_embedding(self, semantic_features: Dict[str, Any]) -> List[float]:
        """Generate semantic embedding (placeholder implementation)"""
        # In real implementation, use transformer models like FinBERT
        # This is a simplified mock embedding
        
        text_content = " ".join([
            semantic_features.get("primary_text", ""),
            " ".join(semantic_features.get("contextual_texts", [])),
            " ".join(semantic_features.get("domain_terms", [])),
            " ".join(semantic_features.get("synonyms", []))
        ])
        
        # Generate mock 384-dimensional embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text_content.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to 384 dimensions
        while len(embedding) < 384:
            embedding.extend(embedding[:384-len(embedding)])
        
        return embedding[:384]

    async def _generate_hierarchical_embedding(self, hierarchical_features: Dict[str, Any]) -> List[float]:
        """Generate hierarchical embedding (128 dimensions)"""
        # Mock hierarchical embedding based on entity type and level
        entity_type_map = {
            "account": 0.1, "location": 0.3, "product": 0.5, 
            "measure": 0.7, "book": 0.9
        }
        
        entity_type = hierarchical_features.get("entity_type", "")
        type_value = entity_type_map.get(entity_type, 0.5)
        
        hierarchy_level = hierarchical_features.get("hierarchy_level", 0)
        level_value = min(1.0, hierarchy_level / 10.0)  # Normalize level
        
        # Generate 128-dimensional embedding
        embedding = []
        for i in range(128):
            if i < 64:
                embedding.append(type_value + (i / 1000.0))
            else:
                embedding.append(level_value + ((i - 64) / 1000.0))
        
        return embedding

    async def _generate_contextual_embedding(self, contextual_features: Dict[str, Any]) -> List[float]:
        """Generate contextual embedding (256 dimensions)"""
        # Mock contextual embedding based on business context
        criticality = contextual_features.get("business_criticality", 0.5)
        complexity = contextual_features.get("regulatory_complexity", 0.3)
        impact = contextual_features.get("stakeholder_impact", 0) / 10.0  # Normalize
        scope = contextual_features.get("operational_scope", 0.5)
        
        base_values = [criticality, complexity, impact, scope]
        
        # Generate 256-dimensional embedding
        embedding = []
        for i in range(256):
            base_idx = i % len(base_values)
            value = base_values[base_idx] + (i / 10000.0)  # Add slight variation
            embedding.append(min(1.0, value))
        
        return embedding

    async def _generate_relationship_embedding(self, relationship_features: Dict[str, Any]) -> List[float]:
        """Generate relationship embedding (192 dimensions)"""
        # Mock relationship embedding
        rel_count = min(1.0, relationship_features.get("relationship_count", 0) / 20.0)
        high_conf = min(1.0, relationship_features.get("high_confidence_relationships", 0) / 10.0)
        cross_type = min(1.0, relationship_features.get("cross_type_connections", 0) / 10.0)
        diversity = min(1.0, relationship_features.get("relationship_diversity", 0) / 5.0)
        
        base_values = [rel_count, high_conf, cross_type, diversity]
        
        # Generate 192-dimensional embedding
        embedding = []
        for i in range(192):
            base_idx = i % len(base_values)
            value = base_values[base_idx] + (i / 5000.0)
            embedding.append(min(1.0, value))
        
        return embedding

    async def _generate_quality_embedding(self, quality_features: Dict[str, Any]) -> List[float]:
        """Generate quality embedding (64 dimensions)"""
        # Mock quality embedding
        completeness = quality_features.get("data_completeness", 0.5)
        enrichment = quality_features.get("enrichment_quality", 0.5)
        relationship = quality_features.get("relationship_quality", 0.5)
        coherence = quality_features.get("semantic_coherence", 0.5)
        
        base_values = [completeness, enrichment, relationship, coherence]
        
        # Generate 64-dimensional embedding
        embedding = []
        for i in range(64):
            base_idx = i % len(base_values)
            value = base_values[base_idx] + (i / 2000.0)
            embedding.append(min(1.0, value))
        
        return embedding

    async def _generate_temporal_embedding(self, temporal_features: Dict[str, Any]) -> List[float]:
        """Generate temporal embedding (96 dimensions)"""
        # Mock temporal embedding based on timestamps
        relevance = temporal_features.get("temporal_relevance", 1.0)
        freshness = temporal_features.get("data_freshness", 1.0)
        
        base_values = [relevance, freshness, 0.8, 0.9]  # Add some default temporal values
        
        # Generate 96-dimensional embedding
        embedding = []
        for i in range(96):
            base_idx = i % len(base_values)
            value = base_values[base_idx] + (i / 3000.0)
            embedding.append(min(1.0, value))
        
        return embedding

    async def _generate_composite_embedding(self, embeddings: Dict[str, List[float]]) -> List[float]:
        """Generate composite embedding by weighted combination"""
        # Weights for different embedding types
        weights = {
            "semantic": 0.4,
            "hierarchical": 0.2,
            "contextual": 0.2,
            "relationship": 0.1,
            "quality": 0.05,
            "temporal": 0.05
        }
        
        # Get dimensions
        total_dims = sum(len(emb) for emb in embeddings.values() if isinstance(emb, list))
        
        composite = []
        current_idx = 0
        
        for emb_type, weight in weights.items():
            if emb_type in embeddings and isinstance(embeddings[emb_type], list):
                emb = embeddings[emb_type]
                weighted_emb = [val * weight for val in emb]
                composite.extend(weighted_emb)
        
        return composite

    def _get_single_entity_embedding_dimensions(self, embeddings: Dict[str, List[float]]) -> Dict[str, int]:
        """Get embedding dimensions for a single entity"""
        return {key: len(emb) for key, emb in embeddings.items() if isinstance(emb, list)}

    def _get_model_versions(self) -> Dict[str, str]:
        """Get model version information"""
        return {
            "semantic_model": "financial-bert-v1.0",
            "hierarchical_model": "categorical-embedding-v1.0",
            "contextual_model": "business-context-encoder-v1.0",
            "relationship_model": "graph-embedding-v1.0",
            "quality_model": "quality-assessment-v1.0",
            "temporal_model": "temporal-encoder-v1.0"
        }

    async def _assess_embedding_quality(self, embeddings: Dict[str, List[float]]) -> float:
        """Assess quality of generated embeddings"""
        quality_scores = []
        
        for emb_type, emb in embeddings.items():
            if isinstance(emb, list) and emb:
                # Check for valid range (0-1)
                in_range = all(0 <= val <= 1 for val in emb)
                # Check for diversity (not all same values)
                has_diversity = len(set(emb)) > len(emb) * 0.1
                
                type_quality = 1.0 if in_range and has_diversity else 0.5
                quality_scores.append(type_quality)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    async def _structure_knowledge_graph(self, embedded_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Structure knowledge graph representation"""
        try:
            # Generate RDF triples
            rdf_triples = []
            turtle_lines = ["@prefix finsight: <http://finsight.ai/ontology/> .", ""]
            
            for entity in embedded_entities:
                entity_id = entity.get("entity_id", "")
                entity_uri = f"finsight:entity_{entity_id}"
                
                # Add basic entity triples
                turtle_lines.append(f"{entity_uri} a finsight:FinancialEntity ;")
                turtle_lines.append(f"    finsight:hasId \"{entity_id}\" ;")
                turtle_lines.append(f"    finsight:hasEmbedding finsight:embedding_{entity_id} .")
                turtle_lines.append("")
                
                # Add embedding resource
                embedding_uri = f"finsight:embedding_{entity_id}"
                embeddings = entity.get("embeddings", {})
                
                turtle_lines.append(f"{embedding_uri} a finsight:VectorEmbedding ;")
                for emb_type, emb_vector in embeddings.items():
                    if isinstance(emb_vector, list):
                        turtle_lines.append(f"    finsight:has{emb_type.title()}Embedding \"{','.join(map(str, emb_vector[:5]))}...\" ;")
                turtle_lines.append(f"    finsight:generatedAt \"{datetime.utcnow().isoformat()}\" .")
                turtle_lines.append("")
                
                rdf_triples.extend([
                    (entity_uri, "rdf:type", "finsight:FinancialEntity"),
                    (entity_uri, "finsight:hasId", f'"{entity_id}"'),
                    (entity_uri, "finsight:hasEmbedding", embedding_uri)
                ])
            
            turtle_format = "\n".join(turtle_lines)
            
            # Generate vector index metadata
            vector_index = {
                "index_type": "composite_embedding",
                "dimensions": self._get_embedding_dimensions(embedded_entities),
                "entity_count": len(embedded_entities),
                "index_timestamp": datetime.utcnow().isoformat(),
                "similarity_metric": "cosine"
            }
            
            return {
                "turtle_format": turtle_format,
                "rdf_triples": rdf_triples,
                "triple_count": len(rdf_triples),
                "vector_index": vector_index
            }
            
        except Exception as e:
            logger.error(f"Failed to structure knowledge graph: {e}")
            return {
                "turtle_format": "",
                "rdf_triples": [],
                "triple_count": 0,
                "vector_index": {}
            }

    def _get_embedding_dimensions(self, embedded_entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get embedding dimensions from entities"""
        if not embedded_entities:
            return {}
        
        first_entity = embedded_entities[0]
        embeddings = first_entity.get("embeddings", {})
        
        return {key: len(emb) for key, emb in embeddings.items() if isinstance(emb, list)}

    async def _validate_ai_readiness(self, embedded_entities: List[Dict[str, Any]], knowledge_graph_data: Dict[str, Any]) -> AIReadinessValidationResult:
        """Validate AI readiness of processed entities"""
        try:
            issues = []
            
            # Check entity count
            if len(embedded_entities) == 0:
                issues.append("No entities were successfully processed")
                return AIReadinessValidationResult(
                    ready_for_ai=False,
                    overall_readiness_score=0.0,
                    embedding_quality=0.0,
                    relationship_completeness=0.0,
                    semantic_coherence=0.0,
                    issues=issues,
                    validation_timestamp=datetime.utcnow().isoformat()
                )
            
            # Validate embedding quality
            embedding_qualities = []
            for entity in embedded_entities:
                embedding_metadata = entity.get("embedding_metadata", {})
                quality = embedding_metadata.get("quality_score", 0.0)
                embedding_qualities.append(quality)
            
            avg_embedding_quality = sum(embedding_qualities) / len(embedding_qualities)
            
            if avg_embedding_quality < 0.8:
                issues.append(f"Low embedding quality: {avg_embedding_quality:.2f}")
            
            # Validate relationship completeness
            total_relationships = sum(len(entity.get("relationships", [])) for entity in embedded_entities)
            avg_relationships = total_relationships / len(embedded_entities)
            relationship_completeness = min(1.0, avg_relationships / 5.0)  # Expect ~5 relationships per entity
            
            if relationship_completeness < 0.9:
                issues.append(f"Low relationship completeness: {relationship_completeness:.2f}")
            
            # Validate semantic coherence
            coherence_scores = []
            for entity in embedded_entities:
                # Mock coherence validation
                coherence_scores.append(0.85)  # Placeholder
            
            avg_semantic_coherence = sum(coherence_scores) / len(coherence_scores)
            
            if avg_semantic_coherence < 0.85:
                issues.append(f"Low semantic coherence: {avg_semantic_coherence:.2f}")
            
            # Validate knowledge graph structure
            if knowledge_graph_data.get("triple_count", 0) < len(embedded_entities) * 3:
                issues.append("Insufficient RDF triple generation")
            
            # Calculate overall readiness score
            scores = [avg_embedding_quality, relationship_completeness, avg_semantic_coherence]
            overall_score = sum(scores) / len(scores)
            
            # Determine if ready for AI
            ready_for_ai = len(issues) == 0 and overall_score >= 0.8
            
            return AIReadinessValidationResult(
                ready_for_ai=ready_for_ai,
                overall_readiness_score=overall_score,
                embedding_quality=avg_embedding_quality,
                relationship_completeness=relationship_completeness,
                semantic_coherence=avg_semantic_coherence,
                issues=issues,
                validation_timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"AI readiness validation failed: {e}")
            return AIReadinessValidationResult(
                ready_for_ai=False,
                overall_readiness_score=0.0,
                embedding_quality=0.0,
                relationship_completeness=0.0,
                semantic_coherence=0.0,
                issues=[f"Validation error: {str(e)}"],
                validation_timestamp=datetime.utcnow().isoformat()
            )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status using AgentTaskTracker"""
        return self.task_tracker.get_task_status(task_id)

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status (async version for compatibility)"""
        return self.task_tracker.get_task_status(task_id)

    async def start_message_queue_processor(self):
        """Start the message queue processor"""
        await self.message_queue.start_processor()
        logger.info(f"Message queue processor started for {self.agent_name}")

    async def stop_message_queue_processor(self):
        """Stop the message queue processor"""
        await self.message_queue.stop_processor()
        logger.info(f"Message queue processor stopped for {self.agent_name}")
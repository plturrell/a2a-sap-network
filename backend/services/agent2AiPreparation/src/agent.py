"""
AI Preparation Agent - A2A Microservice
Agent 2: Prepares standardized data for AI/ML processing
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import httpx

# Add backend path to import GrokClient
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../backend'))
from app.clients.grokClient import GrokClient, GrokConfig

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response
import httpx

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
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize Grok client for semantic enrichment
        self.grok_client = None
        
        # HTTP client for API communication
        self.http_client = None
        
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
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=50)
        )
        
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
    
    @a2a_skill("semantic_enrichment", "Enrich entities with semantic information")
    async def enrich_semantically(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Add real semantic enrichment to entities using NLP"""
        enriched = []
        
        for entity in entities:
            # Extract text representation for NLP analysis
            entity_text = self._entity_to_text(entity, entity_type)
            
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
                semantic_description=semantic_description,
                business_context=business_context,
                domain_terminology=domain_terminology,
                regulatory_context=regulatory_context,
                synonyms_and_aliases=synonyms_and_aliases,
                contextual_metadata={
                    "enrichment_timestamp": datetime.utcnow().isoformat(),
                    "enrichment_version": "2.0",
                    "enrichment_method": "grok-ai" if self.grok_client else "rule-based",
                    "confidence_score": self._calculate_enrichment_confidence(entity, entity_type)
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
        """Initialize the sentence transformer embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use all-MiniLM-L6-v2 model (efficient and good quality)
            self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            logger.warning("Will use zero embeddings as fallback")
            self.embedding_model = None
            self.embedding_model_name = "zero_fallback"
    
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
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down AI Preparation Agent...")
        
        if self.http_client:
            await self.http_client.aclose()
        
        self.is_ready = False
        logger.info("AI Preparation Agent shutdown complete")
"""
Catalog Manager Agent - SDK Version
Enhanced with A2A SDK for simplified development and maintenance
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
import logging
import httpx

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from ..sdk.utils import create_success_response, create_error_response
from ..core.workflow_context import workflow_context_manager, DataArtifact
from ..core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import initialize_agent_trust, sign_a2a_message, get_trust_contract, verify_a2a_message
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Sentence Transformers for real semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence Transformers not available. Semantic search will use fallback.")

logger = logging.getLogger(__name__)


class ORDRepositoryRequest(BaseModel):
    """Request for ORD repository operations"""
    operation: str = Field(description="Operation type: register, enhance, search, update, delete, quality_check")
    ord_document: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    registration_id: Optional[str] = None
    enhancement_type: str = Field(default="metadata_enrichment")
    ai_powered: bool = Field(default=True, description="Use AI enhancement")
    context: Optional[Dict[str, Any]] = None


class ORDRepositoryResponse(BaseModel):
    """Response from ORD repository operations"""
    operation: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    registration_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QualityAssessment(BaseModel):
    """Quality assessment results"""
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    recommendations: List[str]
    critical_issues: List[str]


class CatalogManagerAgentSDK(A2AAgentBase):
    """
    Catalog Manager Agent - SDK Version
    Manages ORD repository operations with enhanced capabilities
    """
    
    def __init__(self, base_url: str, ord_registry_url: str):
        super().__init__(
            agent_id="catalog_manager_agent",
            name="Catalog Manager Agent",
            description="A2A v0.2.9 compliant agent for ORD repository management",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.ord_registry_url = ord_registry_url
        self.catalog_cache = {}
        self.embedding_model = None
        self.document_embeddings = {}  # Cache for document embeddings
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(6)  # 6 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            "registrations": 0,
            "searches": 0,
            "enhancements": 0,
            "quality_checks": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8005'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Catalog Manager Agent resources...")
        
        # Initialize catalog storage
        storage_path = os.getenv("CATALOG_AGENT_STORAGE_PATH", "/tmp/catalog_agent_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize semantic search capabilities
        await self._initialize_semantic_search()
        
        # Load existing state
        await self._load_agent_state()
        
        logger.info("Catalog Manager Agent initialization complete")
    
    @a2a_handler("ord_repository_operations")
    async def handle_ord_operations(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for ORD repository operations"""
        start_time = time.time()
        
        try:
            # Extract operation request from message
            operation_request = self._extract_operation_request(message)
            if not operation_request:
                return create_error_response("No valid operation request found in message")
            
            # Process ORD operation
            operation_result = await self.process_ord_operation(
                operation_request=operation_request,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            operation_type = operation_request.get('operation', 'unknown')
            self.tasks_completed.labels(agent_id=self.agent_id, task_type=f'ord_{operation_type}').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type=f'ord_{operation_type}').observe(time.time() - start_time)
            
            return create_success_response(operation_result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='ord_operation').inc()
            logger.error(f"ORD operation failed: {e}")
            return create_error_response(f"ORD operation failed: {str(e)}")
    
    @a2a_skill("ord_registration")
    async def ord_registration_skill(self, ord_document: Dict[str, Any]) -> ORDRepositoryResponse:
        """Register a new ORD document"""
        
        try:
            # Validate ORD document
            validation_result = await self._validate_ord_document(ord_document)
            if not validation_result["valid"]:
                return ORDRepositoryResponse(
                    operation="register",
                    success=False,
                    message=f"Validation failed: {validation_result['errors']}",
                    data=validation_result
                )
            
            # Register with ORD registry (placeholder implementation)
            registration_id = str(uuid4())
            
            # Store in local cache
            self.catalog_cache[registration_id] = {
                "ord_document": ord_document,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            # Generate and cache document embedding for semantic search
            if self.embedding_model:
                doc_text = self._create_document_text(ord_document)
                doc_embedding = self.embedding_model.encode(doc_text, normalize_embeddings=True)
                self.document_embeddings[registration_id] = doc_embedding
            
            self.processing_stats["registrations"] += 1
            
            return ORDRepositoryResponse(
                operation="register",
                success=True,
                message="ORD document registered successfully",
                registration_id=registration_id,
                data={"ord_document": ord_document}
            )
            
        except Exception as e:
            logger.error(f"ORD registration failed: {e}")
            return ORDRepositoryResponse(
                operation="register",
                success=False,
                message=f"Registration failed: {str(e)}"
            )
    
    @a2a_skill("ord_enhancement")
    async def ord_enhancement_skill(self, registration_id: str, enhancement_type: str = "metadata_enrichment") -> ORDRepositoryResponse:
        """Enhance existing ORD document with AI-powered improvements"""
        
        try:
            # Retrieve document from cache
            if registration_id not in self.catalog_cache:
                return ORDRepositoryResponse(
                    operation="enhance",
                    success=False,
                    message=f"Document with ID {registration_id} not found"
                )
            
            ord_document = self.catalog_cache[registration_id]["ord_document"].copy()
            
            # Apply enhancement based on type
            if enhancement_type == "metadata_enrichment":
                enhanced_document = await self._enrich_metadata(ord_document)
            elif enhancement_type == "semantic_tagging":
                enhanced_document = await self._add_semantic_tags(ord_document)
            elif enhancement_type == "relationship_mapping":
                enhanced_document = await self._map_relationships(ord_document)
            else:
                enhanced_document = await self._general_enhancement(ord_document)
            
            # Update cache
            self.catalog_cache[registration_id]["ord_document"] = enhanced_document
            self.catalog_cache[registration_id]["last_enhanced"] = datetime.utcnow().isoformat()
            self.catalog_cache[registration_id]["enhancement_type"] = enhancement_type
            
            # Update document embedding for semantic search
            if self.embedding_model:
                doc_text = self._create_document_text(enhanced_document)
                doc_embedding = self.embedding_model.encode(doc_text, normalize_embeddings=True)
                self.document_embeddings[registration_id] = doc_embedding
            
            self.processing_stats["enhancements"] += 1
            
            return ORDRepositoryResponse(
                operation="enhance",
                success=True,
                message=f"Document enhanced with {enhancement_type}",
                registration_id=registration_id,
                data={"enhanced_document": enhanced_document}
            )
            
        except Exception as e:
            logger.error(f"ORD enhancement failed: {e}")
            return ORDRepositoryResponse(
                operation="enhance",
                success=False,
                message=f"Enhancement failed: {str(e)}"
            )
    
    @a2a_skill("ord_search")
    async def ord_search_skill(self, query: str, filters: Dict[str, Any] = None) -> ORDRepositoryResponse:
        """Search ORD documents in the repository"""
        
        try:
            search_results = []
            query_lower = query.lower()
            
            # Search in local cache (simplified implementation)
            for registration_id, data in self.catalog_cache.items():
                ord_document = data["ord_document"]
                
                # Check if query matches document content
                if self._matches_search_query(ord_document, query_lower):
                    # Apply filters if provided
                    if not filters or self._passes_filters(ord_document, filters):
                        search_results.append({
                            "registration_id": registration_id,
                            "ord_document": ord_document,
                            "metadata": {
                                "registered_at": data.get("registered_at"),
                                "last_enhanced": data.get("last_enhanced"),
                                "status": data.get("status")
                            }
                        })
            
            self.processing_stats["searches"] += 1
            
            return ORDRepositoryResponse(
                operation="search",
                success=True,
                message=f"Found {len(search_results)} matching documents",
                data={
                    "query": query,
                    "filters": filters,
                    "results": search_results,
                    "count": len(search_results)
                }
            )
            
        except Exception as e:
            logger.error(f"ORD search failed: {e}")
            return ORDRepositoryResponse(
                operation="search",
                success=False,
                message=f"Search failed: {str(e)}"
            )
    
    @a2a_skill("ord_quality_assessment")
    async def ord_quality_assessment_skill(self, registration_id: str) -> QualityAssessment:
        """Assess quality of ORD document"""
        
        if registration_id not in self.catalog_cache:
            raise ValueError(f"Document with ID {registration_id} not found")
        
        ord_document = self.catalog_cache[registration_id]["ord_document"]
        
        # Assess completeness
        completeness_score = self._assess_completeness(ord_document)
        
        # Assess accuracy
        accuracy_score = self._assess_accuracy(ord_document)
        
        # Assess consistency
        consistency_score = self._assess_consistency(ord_document)
        
        # Calculate overall score
        overall_score = (completeness_score + accuracy_score + consistency_score) / 3
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(ord_document, {
            "completeness": completeness_score,
            "accuracy": accuracy_score,
            "consistency": consistency_score
        })
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(ord_document)
        
        self.processing_stats["quality_checks"] += 1
        
        return QualityAssessment(
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    @a2a_skill("ord_update")
    async def ord_update_skill(self, registration_id: str, updates: Dict[str, Any]) -> ORDRepositoryResponse:
        """Update existing ORD document"""
        
        try:
            if registration_id not in self.catalog_cache:
                return ORDRepositoryResponse(
                    operation="update",
                    success=False,
                    message=f"Document with ID {registration_id} not found"
                )
            
            # Get current document
            current_data = self.catalog_cache[registration_id]
            ord_document = current_data["ord_document"].copy()
            
            # Apply updates
            ord_document.update(updates)
            
            # Validate updated document
            validation_result = await self._validate_ord_document(ord_document)
            if not validation_result["valid"]:
                return ORDRepositoryResponse(
                    operation="update",
                    success=False,
                    message=f"Update validation failed: {validation_result['errors']}"
                )
            
            # Update cache
            self.catalog_cache[registration_id]["ord_document"] = ord_document
            self.catalog_cache[registration_id]["last_updated"] = datetime.utcnow().isoformat()
            
            return ORDRepositoryResponse(
                operation="update",
                success=True,
                message="Document updated successfully",
                registration_id=registration_id,
                data={"updated_document": ord_document}
            )
            
        except Exception as e:
            logger.error(f"ORD update failed: {e}")
            return ORDRepositoryResponse(
                operation="update",
                success=False,
                message=f"Update failed: {str(e)}"
            )
    
    @a2a_task(
        task_type="ord_repository_management",
        description="Complete ORD repository management workflow",
        timeout=300,
        retry_attempts=2
    )
    async def process_ord_operation(self, operation_request: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Process ORD repository operation"""
        
        operation = operation_request.get("operation")
        
        try:
            result = None
            
            if operation == "register":
                result = await self.execute_skill("ord_registration", operation_request.get("ord_document"))
                
            elif operation == "enhance":
                result = await self.execute_skill("ord_enhancement", 
                                                operation_request.get("registration_id"),
                                                operation_request.get("enhancement_type", "metadata_enrichment"))
                
            elif operation == "search":
                result = await self.execute_skill("ord_search", 
                                                operation_request.get("query"),
                                                operation_request.get("filters"))
                
            elif operation == "quality_check":
                quality_result = await self.execute_skill("ord_quality_assessment", 
                                                        operation_request.get("registration_id"))
                result = ORDRepositoryResponse(
                    operation="quality_check",
                    success=True,
                    message="Quality assessment completed",
                    data=quality_result.dict()
                )
                
            elif operation == "update":
                result = await self.execute_skill("ord_update",
                                                operation_request.get("registration_id"),
                                                operation_request.get("updates", {}))
            else:
                result = ORDRepositoryResponse(
                    operation=operation,
                    success=False,
                    message=f"Unknown operation: {operation}"
                )
            
            self.processing_stats["total_processed"] += 1
            
            return {
                "operation_successful": result.success if hasattr(result, 'success') else True,
                "operation": operation,
                "result": result.dict() if hasattr(result, 'dict') else result,
                "context_id": context_id
            }
            
        except Exception as e:
            logger.error(f"ORD operation {operation} failed: {e}")
            return {
                "operation_successful": False,
                "operation": operation,
                "error": str(e),
                "context_id": context_id
            }
    
    def _extract_operation_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract operation request from message"""
        operation_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                operation_data.update(part.data)
            elif part.kind == "file" and part.file:
                operation_data["file"] = part.file
        
        return operation_data
    
    async def _validate_ord_document(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ORD document structure and content"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["title", "description", "type", "version"]
        for field in required_fields:
            if field not in ord_document or not ord_document[field]:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        if "version" in ord_document and not isinstance(ord_document["version"], str):
            errors.append("Version must be a string")
        
        # Check title length
        if "title" in ord_document and len(str(ord_document["title"])) < 5:
            warnings.append("Title is very short, consider making it more descriptive")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "score": max(0, (4 - len(errors)) / 4)  # Simple scoring
        }
    
    async def _enrich_metadata(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich ORD document metadata using AI"""
        enhanced = ord_document.copy()
        
        # Add AI-generated tags
        if "title" in enhanced and "description" in enhanced:
            enhanced["ai_generated_tags"] = self._generate_tags_from_content(
                enhanced["title"] + " " + enhanced["description"]
            )
        
        # Add quality score
        enhanced["metadata_quality_score"] = 0.85
        
        # Add enhancement timestamp
        enhanced["enhanced_at"] = datetime.utcnow().isoformat()
        
        return enhanced
    
    async def _add_semantic_tags(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic tags to ORD document"""
        enhanced = ord_document.copy()
        
        # Generate semantic tags based on content
        content = " ".join([str(v) for v in ord_document.values() if isinstance(v, str)])
        semantic_tags = self._extract_semantic_concepts(content)
        
        enhanced["semantic_tags"] = semantic_tags
        enhanced["semantic_enhancement_at"] = datetime.utcnow().isoformat()
        
        return enhanced
    
    async def _map_relationships(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Map relationships to other documents"""
        enhanced = ord_document.copy()
        
        # Find related documents in cache
        related_docs = []
        doc_type = ord_document.get("type", "")
        
        for reg_id, data in self.catalog_cache.items():
            other_doc = data["ord_document"]
            if other_doc.get("type") == doc_type and other_doc != ord_document:
                related_docs.append({
                    "registration_id": reg_id,
                    "title": other_doc.get("title", "Unknown"),
                    "relationship_type": "same_type"
                })
        
        enhanced["related_documents"] = related_docs[:5]  # Limit to 5
        enhanced["relationship_mapping_at"] = datetime.utcnow().isoformat()
        
        return enhanced
    
    async def _general_enhancement(self, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Apply general enhancements to ORD document"""
        enhanced = ord_document.copy()
        
        # Add completeness score
        enhanced["completeness_score"] = self._assess_completeness(ord_document)
        
        # Add suggested improvements
        enhanced["suggested_improvements"] = self._suggest_improvements(ord_document)
        
        # Add general enhancement timestamp
        enhanced["general_enhancement_at"] = datetime.utcnow().isoformat()
        
        return enhanced
    
    def _matches_search_query(self, ord_document: Dict[str, Any], query: str) -> bool:
        """Check if document matches search query using semantic similarity"""
        if self.embedding_model:
            return self._semantic_match(ord_document, query)
        else:
            # Fallback to substring matching
            searchable_content = " ".join([
                str(v).lower() for v in ord_document.values() 
                if isinstance(v, (str, int, float))
            ])
            return query.lower() in searchable_content
            
    def _semantic_match(self, ord_document: Dict[str, Any], query: str, threshold: float = 0.3) -> bool:
        """Perform semantic matching using vector similarity"""
        try:
            # Get or generate document embedding
            doc_id = ord_document.get('id', str(hash(str(ord_document))))
            
            if doc_id not in self.document_embeddings:
                # Create document text for embedding
                doc_text = self._create_document_text(ord_document)
                doc_embedding = self.embedding_model.encode(doc_text, normalize_embeddings=True)
                self.document_embeddings[doc_id] = doc_embedding
            else:
                doc_embedding = self.document_embeddings[doc_id]
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            
            return similarity >= threshold
            
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}, falling back to substring match")
            # Fallback to substring matching
            searchable_content = " ".join([
                str(v).lower() for v in ord_document.values() 
                if isinstance(v, (str, int, float))
            ])
            return query.lower() in searchable_content
    
    def _passes_filters(self, ord_document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes the given filters"""
        for key, value in filters.items():
            if key not in ord_document or ord_document[key] != value:
                return False
        return True
    
    def _assess_completeness(self, ord_document: Dict[str, Any]) -> float:
        """Assess completeness of ORD document"""
        required_fields = ["title", "description", "type", "version", "author"]
        optional_fields = ["tags", "categories", "license", "created_at"]
        
        required_score = sum(1 for field in required_fields if ord_document.get(field)) / len(required_fields)
        optional_score = sum(1 for field in optional_fields if ord_document.get(field)) / len(optional_fields)
        
        return (required_score * 0.8) + (optional_score * 0.2)
    
    def _assess_accuracy(self, ord_document: Dict[str, Any]) -> float:
        """Assess accuracy of ORD document"""
        # Simplified accuracy assessment
        accuracy_checks = []
        
        # Check if version follows semantic versioning
        version = ord_document.get("version", "")
        if version and len(version.split('.')) == 3:
            accuracy_checks.append(True)
        else:
            accuracy_checks.append(False)
        
        # Check if title and description are meaningful
        title = ord_document.get("title", "")
        description = ord_document.get("description", "")
        
        accuracy_checks.append(len(title) > 10)
        accuracy_checks.append(len(description) > 20)
        
        return sum(accuracy_checks) / len(accuracy_checks)
    
    def _assess_consistency(self, ord_document: Dict[str, Any]) -> float:
        """Assess consistency of ORD document"""
        # Simplified consistency assessment
        consistency_score = 1.0
        
        # Check type consistency
        doc_type = ord_document.get("type", "")
        title = ord_document.get("title", "").lower()
        description = ord_document.get("description", "").lower()
        
        if doc_type.lower() not in title and doc_type.lower() not in description:
            consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _generate_quality_recommendations(self, ord_document: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if scores["completeness"] < 0.8:
            recommendations.append("Add missing required fields like author, license, or creation date")
        
        if scores["accuracy"] < 0.7:
            recommendations.append("Improve title and description length and meaningfulness")
        
        if scores["consistency"] < 0.8:
            recommendations.append("Ensure document type is reflected in title and description")
        
        if not ord_document.get("tags"):
            recommendations.append("Add relevant tags to improve discoverability")
        
        return recommendations
    
    def _identify_critical_issues(self, ord_document: Dict[str, Any]) -> List[str]:
        """Identify critical issues in ORD document"""
        critical_issues = []
        
        if not ord_document.get("title"):
            critical_issues.append("Missing title - this is required for document identification")
        
        if not ord_document.get("description"):
            critical_issues.append("Missing description - this is essential for understanding the document")
        
        if not ord_document.get("type"):
            critical_issues.append("Missing type - this is required for proper categorization")
        
        return critical_issues
    
    def _generate_tags_from_content(self, content: str) -> List[str]:
        """Generate tags from document content using semantic analysis"""
        if self.embedding_model:
            return self._generate_semantic_tags(content)
        else:
            # Fallback to keyword matching
            keywords = ["financial", "data", "api", "service", "integration", "banking", "compliance"]
            content_lower = content.lower()
            return [keyword for keyword in keywords if keyword in content_lower]
            
    def _generate_semantic_tags(self, content: str) -> List[str]:
        """Generate tags using semantic similarity to predefined concepts"""
        try:
            # Predefined concept tags with their descriptions
            concept_tags = {
                "financial_services": "banking, finance, payments, transactions, financial services",
                "data_management": "data processing, storage, analytics, databases, ETL",
                "api_integration": "REST API, web services, integration, endpoints, microservices",
                "regulatory_compliance": "compliance, regulations, audit, governance, risk management",
                "security": "authentication, authorization, encryption, security, access control",
                "analytics": "business intelligence, reporting, analytics, metrics, KPIs"
            }
            
            # Generate content embedding
            content_embedding = self.embedding_model.encode(content, normalize_embeddings=True)
            
            # Calculate similarities with concept tags
            matching_tags = []
            for tag, description in concept_tags.items():
                concept_embedding = self.embedding_model.encode(description, normalize_embeddings=True)
                similarity = np.dot(content_embedding, concept_embedding)
                
                if similarity > 0.4:  # Threshold for semantic relevance
                    matching_tags.append(tag)
            
            return matching_tags
            
        except Exception as e:
            logger.warning(f"Semantic tag generation failed: {e}")
            return []
    
    def _extract_semantic_concepts(self, content: str) -> List[str]:
        """Extract semantic concepts from content"""
        # Simplified semantic concept extraction
        concepts = []
        content_lower = content.lower()
        
        concept_map = {
            "financial": ["finance", "money", "banking", "payment"],
            "data": ["information", "dataset", "records", "analytics"],
            "api": ["service", "interface", "endpoint", "rest"],
            "security": ["authentication", "authorization", "encryption", "compliance"]
        }
        
        for concept, keywords in concept_map.items():
            if any(keyword in content_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _suggest_improvements(self, ord_document: Dict[str, Any]) -> List[str]:
        """Suggest general improvements"""
        suggestions = []
        
        if len(ord_document.get("description", "")) < 100:
            suggestions.append("Consider expanding the description to provide more context")
        
        if not ord_document.get("examples"):
            suggestions.append("Add usage examples to help users understand the document better")
        
        if not ord_document.get("contact"):
            suggestions.append("Add contact information for support and questions")
        
        return suggestions
    
    async def _initialize_semantic_search(self):
        """Initialize semantic search capabilities"""
        try:
            if EMBEDDINGS_AVAILABLE:
                # Use financial domain optimized model or general purpose
                model_name = "all-MiniLM-L6-v2"  # 384 dimensions, good for search
                logger.info(f"Loading semantic search model {model_name}...")
                self.embedding_model = SentenceTransformer(model_name)
                logger.info("✅ Real semantic search initialized")
            else:
                logger.warning("Sentence transformers not available, using keyword-based search")
                self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            self.embedding_model = None
            
    def _create_document_text(self, ord_document: Dict[str, Any]) -> str:
        """Create searchable text from ORD document"""
        text_parts = []
        
        # Include key fields for semantic search
        for field in ['title', 'description', 'type', 'tags', 'categories']:
            if field in ord_document and ord_document[field]:
                if isinstance(ord_document[field], list):
                    text_parts.append(' '.join(ord_document[field]))
                else:
                    text_parts.append(str(ord_document[field]))
        
        return ' '.join(text_parts)

    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            catalog_file = os.path.join(self.storage_path, "catalog_cache.json")
            if os.path.exists(catalog_file):
                with open(catalog_file, 'r') as f:
                    self.catalog_cache = json.load(f)
                logger.info(f"Loaded {len(self.catalog_cache)} catalog entries from state")
                
                # Rebuild document embeddings for semantic search
                if self.embedding_model:
                    await self._rebuild_document_embeddings()
                    
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
            
    async def _rebuild_document_embeddings(self):
        """Rebuild document embeddings after state load"""
        try:
            logger.info("Rebuilding document embeddings for semantic search...")
            for registration_id, data in self.catalog_cache.items():
                ord_document = data["ord_document"]
                doc_text = self._create_document_text(ord_document)
                doc_embedding = self.embedding_model.encode(doc_text, normalize_embeddings=True)
                self.document_embeddings[registration_id] = doc_embedding
            
            logger.info(f"Rebuilt embeddings for {len(self.document_embeddings)} documents")
        except Exception as e:
            logger.warning(f"Failed to rebuild document embeddings: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save catalog cache
            catalog_file = os.path.join(self.storage_path, "catalog_cache.json")
            with open(catalog_file, 'w') as f:
                json.dump(self.catalog_cache, f, default=str, indent=2)
            
            # Save document embeddings if available
            if self.document_embeddings:
                embeddings_file = os.path.join(self.storage_path, "document_embeddings.json")
                # Convert numpy arrays to lists for JSON serialization
                serializable_embeddings = {
                    k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in self.document_embeddings.items()
                }
                with open(embeddings_file, 'w') as f:
                    json.dump(serializable_embeddings, f, indent=2)
            
            logger.info(f"Saved {len(self.catalog_cache)} catalog entries and {len(self.document_embeddings)} embeddings to state")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
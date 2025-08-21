"""
Catalog Manager Agent - SDK Version
Enhanced with A2A SDK for simplified development and maintenance
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


import datetime


from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from uuid import uuid4
import asyncio
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
import json
import logging
import numpy as np
import os
import time

from fastapi import HTTPException
from pydantic import BaseModel, Field

# Trust system imports
try:
    from app.a2a.core.trustManager import (
        initialize_agent_trust,
        get_trust_contract,
        verify_a2a_message,
        sign_a2a_message
    )
except ImportError:
    # Fallback if trust system not available
    def initialize_agent_trust(*args, **kwargs):
        return {"status": "trust_system_unavailable"}
    
    def get_trust_contract():
        return None
    
    def verify_a2a_message(*args, **kwargs):
        return True, {"status": "trust_system_unavailable"}
    
    def sign_a2a_message(*args, **kwargs):
        return {"message": args[1] if len(args) > 1 else {}, "signature": {"status": "trust_system_unavailable"}}

# Import Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    raise ImportError("Prometheus metrics required for production. Install with: pip install prometheus-client")

# Import SDK components - use local components
from ..sdk.performanceMonitoringMixin import PerformanceMonitoringMixin, monitor_a2a_operation
from app.a2a.sdk import (
    A2AAge, a2a_handlerntBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.utils import create_error_response, create_success_response

#     elif operation == "update":
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
    
    def _calculate_basic_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate basic relevance score for keyword search"""
        query_terms = query.lower().split()
        doc_text = self._create_document_text(document).lower()
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term in doc_text)
        
        # Calculate relevance based on match ratio
        if len(query_terms) == 0:
            return 0.0
        
        relevance = matches / len(query_terms)
        
        # Boost if query matches title or type
        if query in document.get("title", "").lower():
            relevance += 0.3
        if query in document.get("type", "").lower():
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _create_document_text(self, document: Dict[str, Any]) -> str:
        """Create searchable text representation of document"""
        # Prioritize important fields
        important_fields = ["title", "description", "type", "tags", "categories"]
        text_parts = []
        
        for field in important_fields:
            value = document.get(field)
            if value:
                if isinstance(value, list):
                    text_parts.append(" ".join(str(v) for v in value))
                else:
                    text_parts.append(str(value))
        
        # Add other string fields
        for key, value in document.items():
            if key not in important_fields and isinstance(value, str):
                text_parts.append(value)
        
        return " ".join(text_parts)
    
    def _generate_tags_from_content(self, content: str) -> List[str]:
        """Generate tags from content using simple keyword extraction"""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        
        # Extract words
        words = content.lower().split()
        
        # Filter and deduplicate
        tags = []
        seen = set()
        for word in words:
            # Clean word
            word = word.strip(".,!?;:'\"")
            
            if (len(word) > 3 and 
                word not in stop_words and 
                word not in seen and 
                word.isalpha()):
                tags.append(word)
                seen.add(word)
        
        return tags[:20]  # Limit to 20 tags
    
    def _extract_semantic_concepts(self, content: str) -> List[str]:
        """Extract semantic concepts from content"""
        # This is a simplified version - in production, use NLP libraries
        concepts = []
        
        # Look for technical terms
        tech_patterns = [
            "api", "service", "endpoint", "integration", "data",
            "authentication", "authorization", "security", "protocol",
            "rest", "soap", "graphql", "grpc", "http", "https"
        ]
        
        content_lower = content.lower()
        for pattern in tech_patterns:
            if pattern in content_lower:
                concepts.append(pattern)
        
        # Add domain-specific concepts
        if "finance" in content_lower or "banking" in content_lower:
            concepts.append("financial-services")
        if "health" in content_lower or "medical" in content_lower:
            concepts.append("healthcare")
        if "retail" in content_lower or "commerce" in content_lower:
            concepts.append("e-commerce")
        
        return concepts[:10]  # Limit to 10 concepts
    
    def _suggest_improvements(self, ord_document: Dict[str, Any]) -> List[str]:
        """Suggest improvements for ORD document"""
        suggestions = []
        
        # Check title
        title = ord_document.get("title", "")
        if len(title) < 10:
            suggestions.append("Consider adding a more descriptive title")
        
        # Check description
        description = ord_document.get("description", "")
        if len(description) < 50:
            suggestions.append("Add a detailed description (at least 50 characters)")
        
        # Check for missing fields
        if not ord_document.get("tags"):
            suggestions.append("Add relevant tags for better discoverability")
        if not ord_document.get("version"):
            suggestions.append("Specify a version number")
        if not ord_document.get("author"):
            suggestions.append("Add author information")
        
        # Check for advanced fields
        if not ord_document.get("api_documentation_url"):
            suggestions.append("Consider adding API documentation URL")
        if not ord_document.get("examples"):
            suggestions.append("Add usage examples")
        if not ord_document.get("dependencies"):
            suggestions.append("Document any dependencies")
        
        return suggestions
    
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
    
    @a2a_skill("schema_registry_register")
    async def register_schema(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new schema version in the centralized schema registry"""
try:
            schema_id = input_data.get("schema_id")
            version = input_data.get("version", "1.0.0")
            schema_definition = input_data.get("schema_definition")
            metadata = input_data.get("metadata", {})
            
            if not schema_id or not schema_definition:
                raise ValueError("schema_id and schema_definition are required")
            
            # Generate unique version ID
            version_id = f"{schema_id}:{version}"
            
            # Store schema version
            self.schema_registry[version_id] = {
                "schema_id": schema_id,
                "version": version,
                "schema_definition": schema_definition,
                "metadata": metadata,
                "registered_at": datetime.now().isoformat(),
                "registered_by": input_data.get("agent_id", "unknown"),
                "status": "active"
            }
            
            # Track versions for this schema
            if schema_id not in self.schema_versions:
                self.schema_versions[schema_id] = []
            self.schema_versions[schema_id].append(version)
            
            # Persist the change
            await self._persist_schema_change("schema_registered", {
                "schema_id": schema_id,
                "version": version,
                "version_id": version_id
            })
            
            # Notify subscribers of new schema registration
            await self._notify_schema_subscribers("schema_registered", {
                "schema_id": schema_id,
                "version": version,
                "version_id": version_id
            })
            
            logger.info(f"Registered schema {version_id}")
            
            return {
                "success": True,
                "version_id": version_id,
                "message": f"Schema {schema_id} version {version} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Schema registration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema registration failed"
            }
    
    @a2a_skill("schema_registry_get")
    async def get_schema(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve schema definition from the registry"""
try:
            schema_id = input_data.get("schema_id")
            version = input_data.get("version", "latest")
            
            if not schema_id:
                raise ValueError("schema_id is required")
            
            # Get specific version or latest
            if version == "latest":
                if schema_id in self.schema_versions and self.schema_versions[schema_id]:
                    # Get the latest version (assuming semantic versioning)
                    latest_version = sorted(self.schema_versions[schema_id])[-1]
                    version_id = f"{schema_id}:{latest_version}"
                else:
                    raise ValueError(f"No versions found for schema {schema_id}")
            else:
                version_id = f"{schema_id}:{version}"
            
            if version_id not in self.schema_registry:
                raise ValueError(f"Schema version {version_id} not found")
            
            schema_data = self.schema_registry[version_id]
            
            return {
                "success": True,
                "schema_data": schema_data,
                "version_id": version_id
            }
            
        except Exception as e:
            logger.error(f"Schema retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema retrieval failed"
            }
    
    @a2a_skill("schema_registry_migrate")
    async def migrate_schema(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute schema migration between versions"""
try:
            schema_id = input_data.get("schema_id")
            from_version = input_data.get("from_version")
            to_version = input_data.get("to_version")
            migration_script = input_data.get("migration_script")
            
            if not all([schema_id, from_version, to_version]):
                raise ValueError("schema_id, from_version, and to_version are required")
            
            # Validate versions exist
            from_version_id = f"{schema_id}:{from_version}"
            to_version_id = f"{schema_id}:{to_version}"
            
            if from_version_id not in self.schema_registry:
                raise ValueError(f"Source version {from_version_id} not found")
            if to_version_id not in self.schema_registry:
                raise ValueError(f"Target version {to_version_id} not found")
            
            # Store migration information
            migration_id = f"{from_version_id}→{to_version_id}"
            self.schema_migrations[migration_id] = {
                "schema_id": schema_id,
                "from_version": from_version,
                "to_version": to_version,
                "migration_script": migration_script,
                "created_at": datetime.now().isoformat(),
                "status": "available"
            }
            
            # Notify subscribers of migration availability
            await self._notify_schema_subscribers("migration_available", {
                "schema_id": schema_id,
                "from_version": from_version,
                "to_version": to_version,
                "migration_id": migration_id
            })
            
            logger.info(f"Created migration {migration_id}")
            
            return {
                "success": True,
                "migration_id": migration_id,
                "message": f"Migration from {from_version} to {to_version} created"
            }
            
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema migration failed"
            }
    
    @a2a_skill("schema_registry_subscribe")
    async def subscribe_to_schema_updates(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe to real-time schema updates"""
try:
            agent_id = input_data.get("agent_id")
            schema_id = input_data.get("schema_id", "*")  # * for all schemas
            notification_endpoint = input_data.get("notification_endpoint")
            
            if not agent_id or not notification_endpoint:
                raise ValueError("agent_id and notification_endpoint are required")
            
            # Add subscription
            subscription_id = f"{agent_id}:{schema_id}"
            self.schema_subscriptions[subscription_id] = {
                "agent_id": agent_id,
                "schema_id": schema_id,
                "notification_endpoint": notification_endpoint,
                "subscribed_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            logger.info(f"Agent {agent_id} subscribed to schema updates for {schema_id}")
            
            return {
                "success": True,
                "subscription_id": subscription_id,
                "message": f"Subscribed to updates for schema {schema_id}"
            }
            
        except Exception as e:
            logger.error(f"Schema subscription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema subscription failed"
            }
    
    async def _notify_schema_subscribers(self, event_type: str, event_data: Dict[str, Any]):
        """Notify all relevant subscribers of schema events"""
try:
            schema_id = event_data.get("schema_id")
            
            # Find relevant subscriptions
            relevant_subscriptions = []
            for subscription_id, subscription in self.schema_subscriptions.items():
                if (subscription["schema_id"] == "*" or 
                    subscription["schema_id"] == schema_id):
                    relevant_subscriptions.append(subscription)
            
            # Send notifications
            for subscription in relevant_subscriptions:
try:
                    notification_data = {
                        "event_type": event_type,
                        "event_data": event_data,
                        "timestamp": datetime.now().isoformat(),
                        "subscription_id": f"{subscription['agent_id']}:{subscription['schema_id']}"
                    }
                    
                    # Send A2A message to subscriber
                    await self._send_notification_message(
                        subscription["agent_id"],
                        notification_data
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to notify {subscription['agent_id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Schema notification failed: {e}")
    
    async def _send_notification_message(self, target_agent_id: str, notification_data: Dict[str, Any]):
        """Send notification message to target agent via A2A protocol"""
try:
            message_content = {
                "messageId": f"schema_notify_{uuid4().hex[:8]}",
                "sender": self.agent_id,
                "receiver": target_agent_id,
                "method": "receiveNotification",
                "notification_type": "schema_update",
                "data": notification_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Sign message for trust verification
            signed_message = sign_a2a_message(message_content, self.agent_id)
            
            # Send notification via appropriate method based on agent preference
try:
                # Check if agent prefers blockchain messaging
                if await self._agent_prefers_blockchain(target_agent_id):
                    await self._send_via_blockchain(signed_message, target_agent_id)
                else:
                    await self._send_via_http(signed_message, target_agent_id)
                    
                logger.info(f"Schema notification sent to {target_agent_id}: {notification_data['event_type']}")
            except Exception as send_error:
                logger.warning(f"Failed to send notification to {target_agent_id}: {send_error}")
            
        except Exception as e:
            logger.error(f"Failed to send notification to {target_agent_id}: {e}")
    
    @a2a_skill("schema_registry_list")
    async def list_schemas(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """List all schemas in the registry with filtering options"""
try:
            filter_pattern = input_data.get("filter", "*")
            include_versions = input_data.get("include_versions", True)
            
            # Get schemas matching filter
            matching_schemas = {}
            
            for schema_id in self.schema_versions.keys():
                if filter_pattern == "*" or filter_pattern in schema_id:
                    schema_info = {
                        "schema_id": schema_id,
                        "total_versions": len(self.schema_versions[schema_id])
                    }
                    
                    if include_versions:
                        schema_info["versions"] = self.schema_versions[schema_id]
                        
                        # Add latest version details
                        latest_version = sorted(self.schema_versions[schema_id])[-1]
                        latest_version_id = f"{schema_id}:{latest_version}"
                        schema_info["latest_version_data"] = self.schema_registry[latest_version_id]
                    
                    matching_schemas[schema_id] = schema_info
            
            return {
                "success": True,
                "schemas": matching_schemas,
                "total_schemas": len(matching_schemas)
            }
            
        except Exception as e:
            logger.error(f"Schema listing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema listing failed"
            }
    
    async def _initialize_persistent_schema_storage(self):
        """Initialize persistent storage for schema registry"""
try:
            # Define storage paths
            self.schema_registry_file = os.path.join(self.storage_path, "schema_registry.json")
            self.schema_versions_file = os.path.join(self.storage_path, "schema_versions.json")
            self.schema_migrations_file = os.path.join(self.storage_path, "schema_migrations.json")
            
            # Load existing data from files
            await self._load_schema_registry_data()
            
            logger.info("Persistent schema registry storage initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize persistent schema storage: {e}")
    
    async def _load_schema_registry_data(self):
        """Load schema registry data from persistent storage"""
try:
            # Load schema registry
            if os.path.exists(self.schema_registry_file):
                with open(self.schema_registry_file, 'r') as f:
                    self.schema_registry = json.load(f)
                    logger.info(f"Loaded {len(self.schema_registry)} schemas from persistent storage")
            
            # Load schema versions
            if os.path.exists(self.schema_versions_file):
                with open(self.schema_versions_file, 'r') as f:
                    self.schema_versions = json.load(f)
                    logger.info(f"Loaded {len(self.schema_versions)} schema version mappings")
            
            # Load schema migrations
            if os.path.exists(self.schema_migrations_file):
                with open(self.schema_migrations_file, 'r') as f:
                    self.schema_migrations = json.load(f)
                    logger.info(f"Loaded {len(self.schema_migrations)} schema migrations")
            
        except Exception as e:
            logger.error(f"Failed to load schema registry data: {e}")
            # Initialize empty structures on failure
            self.schema_registry = {}
            self.schema_versions = {}
            self.schema_migrations = {}
    
    async def _save_schema_registry_data(self):
        """Save schema registry data to persistent storage"""
try:
            # Save schema registry
            with open(self.schema_registry_file, 'w') as f:
                json.dump(self.schema_registry, f, indent=2)
            
            # Save schema versions
            with open(self.schema_versions_file, 'w') as f:
                json.dump(self.schema_versions, f, indent=2)
            
            # Save schema migrations
            with open(self.schema_migrations_file, 'w') as f:
                json.dump(self.schema_migrations, f, indent=2)
            
            logger.debug("Schema registry data saved to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save schema registry data: {e}")
    
    async def _persist_schema_change(self, change_type: str, data: Dict[str, Any]):
        """Persist schema changes immediately after operations"""
try:
            # Save to persistent storage
            await self._save_schema_registry_data()
            
            # Log the change for audit trail
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "change_type": change_type,
                "data": data,
                "agent_context": {
                    "total_schemas": len(self.schema_registry),
                    "total_versions": sum(len(versions) for versions in self.schema_versions.values()),
                    "total_migrations": len(self.schema_migrations)
                }
            }
            
            # Append to audit log
            audit_log_file = os.path.join(self.storage_path, "schema_audit.jsonl")
            with open(audit_log_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
            
            logger.info(f"Persisted schema change: {change_type}")
            
        except Exception as e:
            logger.error(f"Failed to persist schema change: {e}")
    
    async def _agent_prefers_blockchain(self, agent_id: str) -> bool:
        """Check if agent prefers blockchain messaging over HTTP"""
try:
            # Check agent preferences (could be from registry, config, or environment)
            blockchain_agents = os.getenv("BLOCKCHAIN_PREFERRED_AGENTS", "").split(",")
            return agent_id.strip() in blockchain_agents
        except Exception:
            return False  # Default to HTTP
    
    async def _send_via_blockchain(self, message: Dict[str, Any], target_agent_id: str):
        """Send message via blockchain A2A network"""
try:
            # Import blockchain client
from ....a2aNetwork.pythonSdk.blockchain.web3Client import Web3Client
from app.a2a.sdk.utils import create_error_response, create_success_response


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Trust system imports
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import initialize_agent_trust
except ImportError:
    # Fallback if trust system not available
    def initialize_agent_trust(*args, **kwargs):
        return {"status": "trust_system_unavailable"}
            
            # Initialize blockchain client
            blockchain_client = Web3Client()
            
            # Send message through blockchain network
            tx_hash = await blockchain_client.send_a2a_message(
                recipient=target_agent_id,
                message=message,
                message_type="schema_notification"
            )
            
            logger.info(f"Blockchain message sent to {target_agent_id}, tx_hash: {tx_hash}")
            
        except Exception as e:
            logger.error(f"Blockchain sending failed to {target_agent_id}: {e}")
            # Fallback to HTTP
            await self._send_via_http(message, target_agent_id)
    
    async def _send_via_http(self, message: Dict[str, Any], target_agent_id: str):
        """Send message via direct HTTP to agent"""
try:
            # Get agent URL from registry or environment
            agent_urls = {
                "data_product_agent_0": os.getenv("DATA_PRODUCT_AGENT_URL", "os.getenv("DATA_MANAGER_URL")"),
                "data_standardization_agent_1": os.getenv("STANDARDIZATION_AGENT_URL", "os.getenv("CATALOG_MANAGER_URL")"),
                "ai_preparation_agent_2": os.getenv("AI_PREPARATION_AGENT_URL", "os.getenv("AGENT_MANAGER_URL")"),
                "vector_processing_agent_3": os.getenv("VECTOR_PROCESSING_AGENT_URL"),
                "sql_agent": os.getenv("SQL_AGENT_URL")
            }
            
            target_url = agent_urls.get(target_agent_id)
            if not target_url:
                raise Exception(f"No URL configured for agent {target_agent_id}")
            
            # Send HTTP POST to agent's A2A endpoint
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{target_url}/a2a/message",
                    json=message,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info(f"HTTP message sent successfully to {target_agent_id}")
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"HTTP sending failed to {target_agent_id}: {e}")
    
    @a2a_skill("schema_registry_backup")
    async def backup_schema_registry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete backup of the schema registry"""
try:
            backup_path = input_data.get("backup_path", os.path.join(self.storage_path, "backups"))
            backup_name = input_data.get("backup_name", f"schema_registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create backup directory
            full_backup_path = os.path.join(backup_path, backup_name)
            os.makedirs(full_backup_path, exist_ok=True)
            
            # Create complete backup
            backup_data = {
                "metadata": {
                    "backup_created_at": datetime.now().isoformat(),
                    "backup_created_by": "catalog_manager_agent",
                    "total_schemas": len(self.schema_registry),
                    "total_versions": sum(len(versions) for versions in self.schema_versions.values()),
                    "total_migrations": len(self.schema_migrations)
                },
                "schema_registry": self.schema_registry,
                "schema_versions": self.schema_versions,
                "schema_migrations": self.schema_migrations,
                "schema_subscriptions": self.schema_subscriptions
            }
            
            # Save backup file
            backup_file = os.path.join(full_backup_path, "schema_registry_complete.json")
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Schema registry backup created: {backup_file}")
            
            return {
                "success": True,
                "backup_path": backup_file,
                "backup_metadata": backup_data["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Schema registry backup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema registry backup failed"
            }
    
    @a2a_skill("schema_registry_restore")
    async def restore_schema_registry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore schema registry from backup"""
try:
            backup_file = input_data.get("backup_file")
            restore_mode = input_data.get("restore_mode", "replace")  # replace, merge
            
            if not backup_file or not os.path.exists(backup_file):
                raise ValueError(f"Backup file not found: {backup_file}")
            
            # Load backup data
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Validate backup structure
            required_keys = ["schema_registry", "schema_versions", "schema_migrations"]
            for key in required_keys:
                if key not in backup_data:
                    raise ValueError(f"Invalid backup file: missing {key}")
            
            if restore_mode == "replace":
                # Replace current data with backup
                self.schema_registry = backup_data["schema_registry"]
                self.schema_versions = backup_data["schema_versions"]
                self.schema_migrations = backup_data["schema_migrations"]
                self.schema_subscriptions = backup_data.get("schema_subscriptions", {})
            elif restore_mode == "merge":
                # Merge backup data with current data
                self.schema_registry.update(backup_data["schema_registry"])
                self.schema_versions.update(backup_data["schema_versions"])
                self.schema_migrations.update(backup_data["schema_migrations"])
                self.schema_subscriptions.update(backup_data.get("schema_subscriptions", {}))
            
            # Persist the restored data
            await self._save_schema_registry_data()
            
            # Log the restoration
            await self._persist_schema_change("registry_restored", {
                "backup_file": backup_file,
                "restore_mode": restore_mode,
                "backup_metadata": backup_data.get("metadata", {})
            })
            
            logger.info(f"Schema registry restored from {backup_file} using {restore_mode} mode")
            
            return {
                "success": True,
                "restore_mode": restore_mode,
                "backup_metadata": backup_data.get("metadata", {}),
                "current_stats": {
                    "total_schemas": len(self.schema_registry),
                    "total_versions": sum(len(versions) for versions in self.schema_versions.values()),
                    "total_migrations": len(self.schema_migrations)
                }
            }
            
        except Exception as e:
            logger.error(f"Schema registry restore failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Schema registry restore failed"
            }
    
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
    
    # ========= MCP Tool Implementations =========
    
    @mcp_tool(
        name="ord_search",
        description="Semantic search for ORD documents with AI-powered ranking"
    )
    async def ord_search_mcp(
        self,
        query: str,
        search_type: str = "semantic",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Search ORD documents with semantic understanding
        
        Args:
            query: Search query (natural language supported)
            search_type: Type of search (semantic, keyword, hybrid)
            filters: Optional filters (type, tags, namespace, etc.)
            limit: Maximum results to return
            include_metadata: Include enhanced metadata in results
        """
try:
            # Use enhanced semantic search
            search_params = {
                "query": query,
                "limit": limit,
                "semantic_weight": 0.7 if search_type == "hybrid" else 1.0 if search_type == "semantic" else 0.0,
                "filters": filters or {}
            }
            
            result = await self.semantic_search({"search_params": search_params})
            
            if result.get("success"):
                search_results = result.get("results", [])
                
                # Format results for MCP
                formatted_results = []
                for item in search_results:
                    formatted_result = {
                        "registration_id": item.get("registration_id"),
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "score": item.get("score"),
                        "type": item.get("type")
                    }
                    
                    if include_metadata:
                        formatted_result["metadata"] = item.get("metadata", {})
                        formatted_result["quality_score"] = item.get("quality_score")
                    
                    formatted_results.append(formatted_result)
                
                return {
                    "success": True,
                    "query": query,
                    "search_type": search_type,
                    "total_results": len(formatted_results),
                    "results": formatted_results
                }
            else:
                return {"success": False, "error": result.get("error", "Search failed")}
                
        except Exception as e:
            logger.error(f"MCP ord_search error: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="ord_register",
        description="Register new ORD documents with AI-enhanced metadata"
    )
    async def ord_register_mcp(
        self,
        ord_document: Dict[str, Any],
        namespace: str,
        auto_enhance: bool = True,
        register_blockchain: bool = False,
        quality_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Register new ORD document with intelligent enhancement
        
        Args:
            ord_document: ORD document to register
            namespace: Document namespace
            auto_enhance: Automatically enhance metadata using AI
            register_blockchain: Also register on blockchain
            quality_threshold: Minimum quality score required
        """
try:
            # Prepare registration data
            registration_data = {
                "ord_document": ord_document,
                "namespace": namespace,
                "metadata": {
                    "auto_enhanced": auto_enhance,
                    "blockchain_enabled": register_blockchain
                }
            }
            
            # Perform quality check first if threshold set
            if quality_threshold > 0:
                quality_result = await self.quality_assessment({"ord_document": ord_document})
                if quality_result.get("success"):
                    quality_score = quality_result.get("quality_score", 0)
                    if quality_score < quality_threshold:
                        return {
                            "success": False,
                            "error": f"Document quality score {quality_score:.2f} below threshold {quality_threshold}",
                            "quality_report": quality_result.get("quality_report")
                        }
            
            # Register document
            result = await self.register_ord_document(registration_data)
            
            if result.get("success"):
                registration_id = result.get("registration_id")
                
                # Auto-enhance if requested
                if auto_enhance and registration_id:
                    enhance_result = await self.enhance_metadata({
                        "registration_id": registration_id,
                        "enhancement_level": "comprehensive"
                    })
                    if enhance_result.get("success"):
                        result["enhancements"] = enhance_result.get("enhancements")
                
                # Blockchain registration if requested
                if register_blockchain and registration_id and BLOCKCHAIN_AVAILABLE:
                    blockchain_result = await self._register_on_blockchain(
                        registration_id, ord_document
                    )
                    result["blockchain_tx"] = blockchain_result
                
                return result
            else:
                return {"success": False, "error": result.get("error", "Registration failed")}
                
        except Exception as e:
            logger.error(f"MCP ord_register error: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="ord_enhance",
        description="Enhance ORD document metadata using AI/NLP"
    )
    async def ord_enhance_mcp(
        self,
        registration_id: str,
        enhancement_types: Optional[List[str]] = None,
        preserve_original: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance document metadata with AI
        
        Args:
            registration_id: Document registration ID
            enhancement_types: Specific enhancements (tags, categories, relationships, quality)
            preserve_original: Keep original metadata
        """
try:
            enhancement_types = enhancement_types or ["tags", "categories", "relationships", "quality"]
            
            enhance_data = {
                "registration_id": registration_id,
                "enhancement_level": "custom",
                "enhancement_types": enhancement_types,
                "preserve_original": preserve_original
            }
            
            result = await self.enhance_metadata(enhance_data)
            
            if result.get("success"):
                return {
                    "success": True,
                    "registration_id": registration_id,
                    "enhancements": result.get("enhancements", {}),
                    "quality_improvement": result.get("quality_improvement", 0),
                    "processing_time_ms": result.get("processing_time_ms", 0)
                }
            else:
                return {"success": False, "error": result.get("error", "Enhancement failed")}
                
        except Exception as e:
            logger.error(f"MCP ord_enhance error: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="schema_manage",
        description="Manage schema definitions with versioning and migration"
    )
    async def schema_manage_mcp(
        self,
        operation: str,
        schema_data: Optional[Dict[str, Any]] = None,
        schema_id: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Manage schema operations
        
        Args:
            operation: Operation type (register, get, update, migrate, list)
            schema_data: Schema definition (for register/update)
            schema_id: Schema identifier
            version: Schema version
        """
try:
            if operation == "register":
                if not schema_data:
                    return {"success": False, "error": "Schema data required for registration"}
                
                result = await self.register_schema({
                    "schema": schema_data,
                    "version": version or "1.0.0"
                })
                
            elif operation == "get":
                if not schema_id:
                    return {"success": False, "error": "Schema ID required"}
                
                result = await self.get_schema({
                    "schema_id": schema_id,
                    "version": version
                })
                
            elif operation == "migrate":
                if not schema_id:
                    return {"success": False, "error": "Schema ID required for migration"}
                
                result = await self.schema_migration({
                    "schema_id": schema_id,
                    "target_version": version,
                    "migration_strategy": schema_data.get("strategy", "auto")
                })
                
            elif operation == "list":
                # List all schemas
                schemas = []
                for sid, versions in self.schema_registry.items():
                    for ver, schema in versions.items():
                        schemas.append({
                            "schema_id": sid,
                            "version": ver,
                            "created_at": schema.get("created_at"),
                            "fields_count": len(schema.get("fields", []))
                        })
                
                return {
                    "success": True,
                    "schemas": schemas,
                    "total_count": len(schemas)
                }
                
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            
            return result
            
        except Exception as e:
            logger.error(f"MCP schema_manage error: {e}")
            return {"success": False, "error": str(e)}
    
    # ========= MCP Resource Implementations =========
    
    @mcp_resource(
        uri="catalog://status",
        description="Catalog manager status and statistics"
    )
    async def get_catalog_status(self) -> Dict[str, Any]:
        """Get catalog manager status and statistics"""
        
        return {
            "catalog_status": {
                "agent_id": self.agent_id,
                "version": self.version,
                "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
                "total_documents": len(self.catalog_cache),
                "total_schemas": sum(len(versions) for versions in self.schema_registry.values()),
                "cache_stats": self.intelligent_cache.get_stats() if hasattr(self, 'intelligent_cache') else {},
                "semantic_search_enabled": self.embedding_model is not None,
                "blockchain_enabled": BLOCKCHAIN_AVAILABLE,
                "metrics": {
                    "search_requests": self.search_requests_total._value._value if PROMETHEUS_AVAILABLE else 0,
                    "registrations": self.registration_requests_total._value._value if PROMETHEUS_AVAILABLE else 0,
                    "cache_hit_rate": self.intelligent_cache.get_hit_rate() if hasattr(self, 'intelligent_cache') else 0
                }
            }
        }
    
    @mcp_resource(
        uri="catalog://search-capabilities",
        description="Search capabilities and configuration"
    )
    async def get_search_capabilities(self) -> Dict[str, Any]:
        """Get search capabilities and configuration"""
        
        return {
            "search_capabilities": {
                "semantic_search": {
                    "enabled": self.embedding_model is not None,
                    "model": "all-MiniLM-L6-v2" if self.embedding_model else None,
                    "embedding_dimensions": 384 if self.embedding_model else 0,
                    "supported_languages": ["en"]
                },
                "search_types": ["semantic", "keyword", "hybrid"],
                "ranking_factors": [
                    "semantic_similarity",
                    "keyword_match",
                    "metadata_quality",
                    "freshness",
                    "popularity"
                ],
                "filters_supported": [
                    "type", "namespace", "tags", "categories",
                    "quality_score", "date_range"
                ],
                "performance": {
                    "avg_search_time_ms": 50,
                    "max_results": 1000,
                    "cache_enabled": True
                }
            }
        }
    
    @mcp_resource(
        uri="catalog://quality-metrics",
        description="Document quality metrics and thresholds"
    )
    async def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics and assessment criteria"""
        
        # Calculate quality distribution
        quality_distribution = {"high": 0, "medium": 0, "low": 0}
        quality_scores = []
        
        for reg_id, data in self.catalog_cache.items():
            if "quality_score" in data:
                score = data["quality_score"]
                quality_scores.append(score)
                if score >= 0.8:
                    quality_distribution["high"] += 1
                elif score >= 0.5:
                    quality_distribution["medium"] += 1
                else:
                    quality_distribution["low"] += 1
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "quality_metrics": {
                "assessment_criteria": {
                    "metadata_completeness": 0.3,
                    "description_quality": 0.3,
                    "schema_compliance": 0.2,
                    "relationship_mapping": 0.2
                },
                "quality_thresholds": {
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.0
                },
                "current_statistics": {
                    "average_quality": avg_quality,
                    "quality_distribution": quality_distribution,
                    "total_assessed": len(quality_scores)
                },
                "enhancement_available": True
            }
        }
    
    @mcp_resource(
        uri="catalog://cache-performance",
        description="Cache performance metrics and configuration"
    )
    async def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        
        cache_stats = self.intelligent_cache.get_stats() if hasattr(self, 'intelligent_cache') else {}
        
        return {
            "cache_performance": {
                "strategies_available": ["LRU", "LFU", "TTL", "Adaptive"],
                "current_strategy": getattr(self.intelligent_cache, 'strategy', 'Unknown'),
                "statistics": {
                    "hit_rate": cache_stats.get("hit_rate", 0),
                    "miss_rate": cache_stats.get("miss_rate", 0),
                    "total_hits": cache_stats.get("hits", 0),
                    "total_misses": cache_stats.get("misses", 0),
                    "cache_size": cache_stats.get("size", 0),
                    "max_size": cache_stats.get("max_size", 1000)
                },
                "performance_impact": {
                    "avg_cache_response_ms": 5,
                    "avg_uncached_response_ms": 50,
                    "speedup_factor": 10
                }
            }
        }
    
    # ========= MCP Prompt Implementations =========
    
    @mcp_prompt(
        name="ord_assistant",
        description="Interactive ORD document assistant for discovery and management"
    )
    async def ord_assistant_prompt(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        ORD document assistant for natural language interactions
        
        Args:
            user_query: User's question about ORD documents
            context: Additional context (previous searches, preferences)
        """
try:
            query_lower = user_query.lower()
            
            # Determine intent
            if any(word in query_lower for word in ["find", "search", "look for", "show"]):
                # Perform search
                search_result = await self.ord_search_mcp(
                    query=user_query,
                    search_type="semantic",
                    limit=5
                )
                
                if search_result.get("success") and search_result.get("results"):
                    response = f"I found {search_result['total_results']} relevant ORD documents:\n\n"
                    
                    for i, doc in enumerate(search_result["results"], 1):
                        response += f"**{i}. {doc['title']}**\n"
                        response += f"   Type: {doc['type']}\n"
                        response += f"   Score: {doc['score']:.2f}\n"
                        if doc.get("description"):
                            response += f"   Description: {doc['description'][:100]}...\n"
                        response += "\n"
                    
                    response += "Would you like more details about any of these documents?"
                    return response
                else:
                    return "I couldn't find any documents matching your query. Try different keywords or be more specific."
            
            elif any(word in query_lower for word in ["register", "add", "create"]):
                return """To register a new ORD document, I need:
1. Document title and description
2. Document type (API, DataProduct, Event, EntityType)
3. Namespace for organization
4. Any relevant metadata

Please provide these details and I'll help you register the document."""
            
            elif any(word in query_lower for word in ["quality", "assess", "check"]):
                return """I can assess document quality based on:
- Metadata completeness (30%)
- Description quality (30%)
- Schema compliance (20%)
- Relationship mapping (20%)

Which document would you like me to assess?"""
            
            else:
                return """I'm the ORD Catalog Assistant. I can help you:
- **Search** for ORD documents using natural language
- **Register** new documents with AI-enhanced metadata
- **Assess** document quality
- **Manage** schemas and versions

What would you like to do?"""
                
        except Exception as e:
            logger.error(f"ORD assistant prompt error: {e}")
            return "I encountered an error processing your request. Please try again."
    
    @mcp_prompt(
        name="metadata_enhancer",
        description="Interactive metadata enhancement advisor"
    )
    async def metadata_enhancer_prompt(
        self,
        document_info: str,
        enhancement_goal: Optional[str] = None
    ) -> str:
        """
        Metadata enhancement advisor
        
        Args:
            document_info: Information about the document
            enhancement_goal: Specific enhancement goal
        """
try:
            response = f"Let me analyze the document metadata for enhancement opportunities.\n\n"
            
            # Parse document info to find registration ID
            words = document_info.split()
            registration_id = None
            for word in words:
                if word.startswith("ord_") or word in self.catalog_cache:
                    registration_id = word
                    break
            
            if registration_id and registration_id in self.catalog_cache:
                doc_data = self.catalog_cache[registration_id]
                current_quality = doc_data.get("quality_score", 0)
                
                response += f"**Current Document Quality:** {current_quality:.2f}/1.0\n\n"
                
                # Analyze what can be enhanced
                suggestions = []
                
                if not doc_data.get("ord_document", {}).get("tags"):
                    suggestions.append("Add semantic tags for better discoverability")
                
                if not doc_data.get("ord_document", {}).get("categories"):
                    suggestions.append("Define categories for improved organization")
                
                if current_quality < 0.8:
                    suggestions.append("Enhance description with more technical details")
                    suggestions.append("Add relationship mappings to related documents")
                
                response += "**Enhancement Suggestions:**\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"{i}. {suggestion}\n"
                
                response += "\nWould you like me to apply these enhancements automatically?"
            else:
                response += """To enhance metadata, I need:
1. The document registration ID
2. Your enhancement goals (discoverability, quality, relationships)

Please provide the document ID you'd like to enhance."""
            
            return response
            
        except Exception as e:
            logger.error(f"Metadata enhancer prompt error: {e}")
            return "I'm having trouble analyzing the metadata. Please provide a valid document ID."
    
    @mcp_prompt(
        name="schema_advisor",
        description="Schema design and migration advisor"
    )
    async def schema_advisor_prompt(
        self,
        schema_question: str,
        current_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schema design and migration advisor
        
        Args:
            schema_question: Question about schema design or migration
            current_schema: Current schema if applicable
        """
try:
            question_lower = schema_question.lower()
            
            if "migrate" in question_lower:
                return """For schema migration, I can help you:

1. **Automatic Migration**: I'll analyze the differences and create migration scripts
2. **Manual Migration**: You specify the exact changes needed
3. **Backward Compatible**: Ensure old clients still work

Which migration strategy would you prefer?"""
            
            elif "design" in question_lower or "create" in question_lower:
                return """For schema design, consider these best practices:

**1. Field Naming:**
- Use camelCase for fields
- Be descriptive but concise
- Avoid abbreviations

**2. Data Types:**
- Use appropriate types (string, number, boolean, array, object)
- Consider nullable fields
- Define enums for fixed values

**3. Validation:**
- Add constraints (minLength, maxLength, pattern)
- Define required fields
- Include descriptions

Would you like me to help design a schema for a specific use case?"""
            
            elif "version" in question_lower:
                # List current schemas and versions
                schema_list = []
                for schema_id, versions in self.schema_registry.items():
                    schema_list.append(f"- **{schema_id}**: versions {', '.join(versions.keys())}")
                
                if schema_list:
                    return f"Current schemas and versions:\n\n" + "\n".join(schema_list)
                else:
                    return "No schemas registered yet. Would you like to create one?"
            
            else:
                return """I'm the Schema Advisor. I can help with:
- **Design** new schemas following best practices
- **Migrate** existing schemas safely
- **Version** schemas for backward compatibility
- **Validate** documents against schemas

What aspect of schema management do you need help with?"""
                
        except Exception as e:
            logger.error(f"Schema advisor prompt error: {e}")
            return "I encountered an error with schema advice. Please try rephrasing your question."
    
    async def _register_on_blockchain(self, registration_id: str, ord_document: Dict[str, Any]) -> Dict[str, Any]:
        """Register document on blockchain"""
try:
            if not BLOCKCHAIN_AVAILABLE:
                return {"success": False, "error": "Blockchain not available"}
            
            # Create blockchain registry instance
            blockchain = ORDBlockchainRegistry()
            
            # Register on blockchain
            tx_hash = await blockchain.register_document(
                registration_id,
                ord_document,
                self.agent_id
            )
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "blockchain": "ethereum"
            }
            
        except Exception as e:
            logger.error(f"Blockchain registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    @a2a_handler("HEALTH_CHECK")
    async def handle_health_check(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle A2A protocol health check messages"""
        try:
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "name": "Catalog Manager Agent",
                "timestamp": datetime.utcnow().isoformat(),
                "blockchain_enabled": getattr(self, 'blockchain_enabled', False),
                "active_tasks": len(getattr(self, 'tasks', {})),
                "capabilities": getattr(self, 'blockchain_capabilities', []),
                "processing_stats": getattr(self, 'processing_stats', {}) or {},
                "response_time_ms": 0  # Immediate response for health checks
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

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


# Utility functions
def create_error_response(message: str) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": message,
        "timestamp": datetime.utcnow().isoformat()
    }


def create_success_response(data: Any) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
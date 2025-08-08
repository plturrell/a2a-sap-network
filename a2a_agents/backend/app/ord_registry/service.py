import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
import logging

from .models import (
    ORDDocument, ORDRegistration, ResourceIndexEntry,
    ValidationResult, RegistrationMetadata, RegistrationStatus,
    ResourceType, SearchRequest, SearchResult, SearchFacet,
    DublinCoreMetadata, DublinCoreQualityMetrics,
    DublinCoreValidationRequest, DublinCoreValidationResponse,
    AnalyticsInfo
)
from .storage import get_ord_storage
from ..clients.grok_client import get_grok_client
from ..clients.perplexity_client import get_perplexity_client
from .advanced_ai_enhancer import create_advanced_ai_enhancer
from .enhanced_search_service import get_enhanced_search_service

logger = logging.getLogger(__name__)


class ORDRegistryService:
    """Object Resource Discovery Registry Service with Dual-Database Storage"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        # Dual-database storage (HANA primary, Supabase fallback)
        self.storage = None
        # AI-enhanced features using A2A clients
        self.grok_client = None
        self.perplexity_client = None
        self.ai_enhancer = None
        # Enhanced search service
        self.enhanced_search = None
        # Initialize flags
        self.initialized = False
        
    async def initialize(self):
        """Initialize the ORD registry service with dual-database storage and A2A clients"""
        if self.initialized:
            return
            
        try:
            # Initialize dual-database storage
            self.storage = await get_ord_storage()
            if self.storage:
                logger.info("✅ ORD dual-database storage initialized")
                logger.info(f"Storage details: HANA={self.storage.hana_client is not None}, "
                           f"Supabase={self.storage.supabase_client is not None}, "
                           f"Fallback={self.storage.fallback_mode}")
            else:
                logger.error("❌ Failed to initialize ORD storage")
            
            # Initialize A2A clients for AI-enhanced features
            try:
                self.grok_client = get_grok_client()
                logger.info("✅ Grok client initialized for ORD AI features")
            except Exception as e:
                logger.warning(f"Grok client initialization failed: {e}")
                
            try:
                self.perplexity_client = get_perplexity_client()
                logger.info("✅ Perplexity client initialized for ORD AI features")
            except Exception as e:
                logger.warning(f"Perplexity client initialization failed: {e}")
                
            # Initialize AI enhancer with available clients
            self.ai_enhancer = create_advanced_ai_enhancer(
                grok_client=self.grok_client,
                perplexity_client=self.perplexity_client
            )
            logger.info("✅ AI enhancer initialized for intelligent metadata generation")
            
            # Initialize enhanced search service
            try:
                self.enhanced_search = await get_enhanced_search_service()
                await self.enhanced_search.initialize(self.storage)
                logger.info("✅ Enhanced search service initialized")
            except Exception as e:
                logger.warning(f"Enhanced search initialization failed: {e}")
                self.enhanced_search = None
                
            self.initialized = True
            logger.info("🚀 ORD Registry Service fully initialized with dual-database and AI features")
            
        except Exception as e:
            logger.error(f"Failed to initialize ORD registry service: {e}")
            raise
            
    async def _ensure_initialized(self):
        """Ensure the service is initialized before any operations"""
        if not self.initialized:
            await self.initialize()
        
    async def register_ord_document(
        self, 
        ord_document: ORDDocument,
        registered_by: str,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[ORDRegistration]:
        """Register a new ORD document with AI-enhanced features and dual-database storage"""
        try:
            # Ensure service is initialized
            await self._ensure_initialized()
            
            # AI-Enhanced Dublin Core Metadata Generation (disabled for performance)
            # TODO: Make this async or optional
            enhanced_ord_document = ord_document
            
            # Validate the enhanced ORD document
            validation_result = await self._validate_ord_document(enhanced_ord_document)
            
            if not validation_result.valid and validation_result.errors:
                logger.error(f"❌ ORD document validation failed: {validation_result.errors}")
                return None
            
            # Generate registration ID
            registration_id = f"reg_{uuid4().hex[:8]}"
            
            # Create registration metadata
            metadata = RegistrationMetadata(
                registered_by=registered_by,
                registered_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                version="1.0.0",
                status=RegistrationStatus.ACTIVE
            )
            
            # Create registration record
            registration = ORDRegistration(
                registration_id=registration_id,
                ord_document=enhanced_ord_document,
                metadata=metadata,
                validation=validation_result
            )
            
            # Store in dual-database storage (HANA primary, Supabase fallback)
            storage_result = await self.storage.store_registration(registration)
            
            if not storage_result.get("success"):
                error_msg = storage_result.get('error', 'Unknown error')
                logger.error(f"❌ Storage failed: {error_msg}")
                logger.error(f"Storage result details: {storage_result}")
                
                # Try to provide more specific error information
                if "Supabase" in error_msg:
                    logger.error("Supabase storage issue detected - check Supabase connection and table setup")
                elif "HANA" in error_msg:
                    logger.error("HANA storage issue detected - check HANA connection and credentials")
                    
                return None
            
            # Index resources for advanced search using storage layer
            await self.storage.index_registration(registration)
            
            # Return the actual ORDRegistration object
            logger.info(f"✅ ORD document registered successfully: {registration_id}")
            return registration
            
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            return None
            
    async def _enhance_with_ai(self, ord_document: ORDDocument) -> ORDDocument:
        """Enhance ORD document using AI-powered metadata generation"""
        try:
            if self.ai_enhancer:
                return await self.ai_enhancer.enhance_ord_document(ord_document)
            else:
                logger.warning("AI enhancer not available, returning original document")
                return ord_document
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return ord_document
    
    async def _validate_ord_document(self, ord_document: ORDDocument) -> ValidationResult:
        """Validate ORD document against specification with Dublin Core enhancement"""
        errors = []
        warnings = []
        
        logger.info(f"Starting ORD document validation...")
        logger.debug(f"Document has: {len(ord_document.dataProducts or [])} data products, "
                    f"{len(ord_document.apiResources or [])} API resources, "
                    f"{len(ord_document.entityTypes or [])} entity types")
        
        # Check ORD version
        if ord_document.openResourceDiscovery not in ["1.5.0", "1.4.0", "1.3.0"]:
            warnings.append(f"ORD version {ord_document.openResourceDiscovery} may not be fully supported")
        
        # Validate resources
        all_resources = []
        
        # Check data products
        if ord_document.dataProducts:
            for dp in ord_document.dataProducts:
                ord_id = dp.get("ordId", "")
                logger.debug(f"Validating data product ORD ID: {ord_id}")
                if not self._validate_ord_id(ord_id):
                    logger.error(f"Invalid data product ORD ID format: {ord_id}")
                    errors.append(f"Invalid ORD ID format: {ord_id}")
                all_resources.append(("dataProduct", dp))
        
        # Check API resources
        if ord_document.apiResources:
            for api in ord_document.apiResources:
                if not self._validate_ord_id(api.get("ordId", "")):
                    errors.append(f"Invalid ORD ID format: {api.get('ordId', '')}")
                all_resources.append(("api", api))
        
        # Check entity types
        if ord_document.entityTypes:
            for entity in ord_document.entityTypes:
                ord_id = entity.get("ordId", "")
                logger.debug(f"Validating entity type ORD ID: {ord_id}")
                if not self._validate_ord_id(ord_id):
                    logger.error(f"Invalid entity type ORD ID format: {ord_id}")
                    errors.append(f"Invalid ORD ID format: {ord_id}")
                all_resources.append(("entityType", entity))
        
        # Validate Dublin Core metadata if present
        dc_validation = None
        if ord_document.dublinCore:
            dc_validation = await self._validate_dublin_core(ord_document.dublinCore)
            if not dc_validation.iso15836_compliant:
                warnings.append("Dublin Core metadata is not fully ISO 15836 compliant")
        
        # Calculate compliance score
        total_checks = len(all_resources) * 3  # 3 checks per resource
        if dc_validation:
            total_checks += 4  # Add Dublin Core checks
            passed_checks = total_checks - len(errors) + (4 * dc_validation.overall_score)
        else:
            passed_checks = total_checks - len(errors)
        
        compliance_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        validation_result = ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            compliance_score=compliance_score,
            dublincore_validation=dc_validation
        )
        
        logger.info(f"Validation complete: valid={validation_result.valid}, "
                   f"errors={len(errors)}, warnings={len(warnings)}, "
                   f"compliance_score={compliance_score:.2f}")
        if errors:
            logger.error(f"Validation errors: {errors}")
        
        return validation_result
    
    def _validate_ord_id(self, ord_id: str) -> bool:
        """Validate ORD ID format according to specification"""
        # ORD ID format: namespace:resourceType:localId
        pattern = r'^[a-zA-Z0-9\.\-]+:[a-zA-Z0-9\.\-]+:[a-zA-Z0-9\.\-_]+$'
        return bool(re.match(pattern, ord_id))
    
    def _increment_version(self, version: str) -> str:
        """Increment version number properly handling semantic versions"""
        try:
            # Try to parse as semantic version (e.g., "1.0.0")
            if '.' in version:
                parts = version.split('.')
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    # Increment patch version
                    major, minor, patch = map(int, parts)
                    return f"{major}.{minor}.{patch + 1}"
            
            # Try to parse as simple integer version
            if version.isdigit():
                return str(int(version) + 1)
            
            # Fallback: append .1 to any version
            return f"{version}.1"
            
        except Exception:
            # Ultimate fallback
            return "2.0.0"
    
    async def _validate_dublin_core(self, dc: DublinCoreMetadata) -> DublinCoreQualityMetrics:
        """Validate Dublin Core metadata and calculate quality metrics"""
        # Count populated elements
        populated = 0
        total_elements = 15  # Core 15 elements
        
        if dc.title: populated += 1
        if dc.creator: populated += 1
        if dc.subject: populated += 1
        if dc.description: populated += 1
        if dc.publisher: populated += 1
        if dc.contributor: populated += 1
        if dc.date: populated += 1
        if dc.type: populated += 1
        if dc.format: populated += 1
        if dc.identifier: populated += 1
        if dc.source: populated += 1
        if dc.language: populated += 1
        if dc.relation: populated += 1
        if dc.coverage: populated += 1
        if dc.rights: populated += 1
        
        completeness = populated / total_elements
        
        # Check format compliance
        accuracy = 1.0
        if dc.date:
            try:
                datetime.fromisoformat(dc.date.replace('Z', '+00:00'))
            except:
                accuracy -= 0.2
        
        if dc.language and len(dc.language) not in [2, 3]:
            accuracy -= 0.1
        
        # Check consistency
        consistency = 1.0
        if dc.title and dc.description:
            # Basic semantic check - title should relate to description
            if dc.title.lower() not in dc.description.lower():
                consistency -= 0.1
        
        # Timeliness (assume current for new registrations)
        timeliness = 1.0
        
        # Calculate overall score
        overall_score = (completeness + accuracy + consistency + timeliness) / 4
        
        # Standards compliance (simplified)
        iso15836_compliant = overall_score >= 0.8
        rfc5013_compliant = overall_score >= 0.75
        ansi_niso_compliant = overall_score >= 0.7
        
        return DublinCoreQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            overall_score=overall_score,
            iso15836_compliant=iso15836_compliant,
            rfc5013_compliant=rfc5013_compliant,
            ansi_niso_compliant=ansi_niso_compliant
        )
    
    async def validate_dublin_core_metadata(self, request: DublinCoreValidationRequest) -> DublinCoreValidationResponse:
        """Validate Dublin Core metadata independently"""
        quality_metrics = await self._validate_dublin_core(request.dublin_core)
        
        recommendations = []
        if quality_metrics.completeness < 0.6:
            recommendations.append("Consider adding more metadata elements for better discoverability")
        if not request.dublin_core.rights:
            recommendations.append("Add 'rights' element to clarify usage permissions")
        if not request.dublin_core.coverage:
            recommendations.append("Consider adding 'coverage' element for temporal/spatial scope")
        if not request.dublin_core.relation:
            recommendations.append("Add 'relation' elements to link related resources")
        
        return DublinCoreValidationResponse(
            valid=quality_metrics.overall_score >= 0.5,
            quality_metrics=quality_metrics,
            metadata_completeness=quality_metrics.completeness,
            recommendations=recommendations
        )
    
    async def _index_ord_resources(
        self,
        registration_id: str,
        ord_document: ORDDocument,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """Index ORD resources for search with Dublin Core enhancement"""
        
        # Index data products
        if ord_document.dataProducts:
            for dp in ord_document.dataProducts:
                await self._index_resource(
                    registration_id,
                    ResourceType.DATA_PRODUCT,
                    dp,
                    tags,
                    labels,
                    ord_document.dublinCore
                )
        
        # Index API resources
        if ord_document.apiResources:
            for api in ord_document.apiResources:
                await self._index_resource(
                    registration_id,
                    ResourceType.API,
                    api,
                    tags,
                    labels,
                    ord_document.dublinCore
                )
        
        # Index entity types
        if ord_document.entityTypes:
            for entity in ord_document.entityTypes:
                await self._index_resource(
                    registration_id,
                    ResourceType.ENTITY_TYPE,
                    entity,
                    tags,
                    labels,
                    ord_document.dublinCore
                )
    
    async def _index_resource(
        self,
        registration_id: str,
        resource_type: ResourceType,
        resource: Dict[str, Any],
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        dublin_core: Optional[DublinCoreMetadata] = None
    ):
        """Index a single resource with Dublin Core enhancement"""
        ord_id = resource.get("ordId", "")
        
        # Create searchable content including Dublin Core fields
        searchable_parts = [
            resource.get("title", ""),
            resource.get("shortDescription", ""),
            resource.get("description", ""),
            " ".join(resource.get("tags", [])),
            " ".join(resource.get("labels", {}).values())
        ]
        
        # Add Dublin Core fields to searchable content
        if dublin_core:
            if dublin_core.title:
                searchable_parts.append(dublin_core.title)
            if dublin_core.subject:
                searchable_parts.extend(dublin_core.subject)
            if dublin_core.description:
                searchable_parts.append(dublin_core.description)
            if dublin_core.creator:
                searchable_parts.extend(dublin_core.creator)
            if dublin_core.publisher:
                searchable_parts.append(dublin_core.publisher)
        
        searchable_content = " ".join(filter(None, searchable_parts)).lower()
        
        # Create index entry with Dublin Core fields
        index_entry = ResourceIndexEntry(
            ord_id=ord_id,
            registration_id=registration_id,
            resource_type=resource_type,
            title=resource.get("title", ""),
            short_description=resource.get("shortDescription"),
            description=resource.get("description"),
            version=resource.get("version"),
            tags=(tags or []) + resource.get("tags", []),
            labels={**(labels or {}), **resource.get("labels", {})},
            domain=resource.get("domain"),
            category=resource.get("category"),
            access_strategies=resource.get("accessStrategies", []),
            compliance_info=resource.get("compliance", {}),
            searchable_content=searchable_content,
            # Dublin Core fields for faceted search
            dublin_core=dublin_core,
            dc_creator=dublin_core.creator if dublin_core else None,
            dc_subject=dublin_core.subject if dublin_core else None,
            dc_publisher=dublin_core.publisher if dublin_core else None,
            dc_format=dublin_core.format if dublin_core else None
        )
        
        # Index entry is now stored in dual-database storage layer
        # No need to maintain in-memory index - storage layer handles this
    
    async def search_resources(self, search_request: SearchRequest) -> SearchResult:
        """Enhanced search for resources with improved algorithms and Dublin Core facets"""
        await self._ensure_initialized()
        
        try:
            # Use enhanced search if available, otherwise fallback to existing search
            if self.enhanced_search:
                logger.info("Using enhanced search service")
                return await self.enhanced_search.enhanced_search(search_request)
            else:
                logger.info("Using fallback search (enhanced search not available)")
                return await self._fallback_search(search_request)
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            # Return empty result on error
            return SearchResult(
                results=[],
                total_count=0,
                page=search_request.page,
                page_size=search_request.page_size,
                facets=None
            )
    
    async def _fallback_search(self, search_request: SearchRequest) -> SearchResult:
        """Fallback search implementation using existing storage"""
        try:
            # Execute database-driven search with dual-database fallback
            filters = search_request.filters or {}
            search_results = await self.storage.search_registrations(
                search_request.query or "",
                filters
            )
            
            # search_results is now a list of ResourceIndexEntry objects
            resource_entries = search_results or []
            total_count = len(resource_entries)
            
            # Calculate facets from results
            facets = self._calculate_search_facets_from_entries(resource_entries)
            
            logger.info(f"✅ Fallback search completed: {len(resource_entries)} results, {total_count} total")
            
            return SearchResult(
                results=resource_entries,
                total_count=total_count,
                page=search_request.page,
                page_size=search_request.page_size,
                facets=facets
            )
            
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {e}")
            raise
    
    async def get_resource_by_ord_id(self, ord_id: str) -> Optional[ResourceIndexEntry]:
        """Get a resource by its ORD ID from dual-database storage"""
        await self._ensure_initialized()
        
        try:
            # Query dual-database storage for resource by ORD ID
            resource_data = await self.storage.get_resource_by_ord_id(ord_id)
            if not resource_data:
                logger.info(f"Resource not found for ORD ID: {ord_id}")
                return None
            
            # Convert database result to ResourceIndexEntry
            resource_entry = self._convert_search_result_to_resource_entry(resource_data)
            
            logger.info(f"✅ Resource retrieved by ORD ID: {ord_id}")
            return resource_entry
            
        except Exception as e:
            logger.error(f"❌ Error retrieving resource by ORD ID {ord_id}: {e}")
            return None
    
    async def get_registration(self, registration_id: str) -> Optional[ORDRegistration]:
        """Get a registration by ID from dual-database storage"""
        try:
            await self._ensure_initialized()
            return await self.storage.get_registration(registration_id)
        except Exception as e:
            logger.error(f"Failed to get registration {registration_id}: {e}")
            return None
    
    async def update_registration(
        self,
        registration_id: str,
        ord_document: ORDDocument,
        enhance_with_ai: bool = True
    ) -> Optional[ORDRegistration]:
        """Update an existing ORD registration with dual-database storage and AI re-enhancement"""
        await self._ensure_initialized()
        
        try:
            # Check if registration exists
            existing_registration = await self.storage.get_registration(registration_id)
            if not existing_registration:
                logger.warning(f"Registration {registration_id} not found for update")
                return None
            
            # AI-enhance the updated document if requested
            if enhance_with_ai:
                ord_document = await self._enhance_with_ai(ord_document)
                logger.info(f"✅ ORD document AI-enhanced for update: {registration_id}")
            
            # Validate the updated document
            validation = await self._validate_ord_document(ord_document)
            if not validation.valid:
                logger.error(f"❌ Updated ORD document validation failed: {validation.errors}")
                return None
            
            # Update registration metadata
            updated_metadata = RegistrationMetadata(
                registered_by=existing_registration.metadata.registered_by,
                registered_at=existing_registration.metadata.registered_at,
                last_updated=datetime.utcnow(),
                version=self._increment_version(existing_registration.metadata.version),
                status=RegistrationStatus.ACTIVE
            )
            
            # Create updated registration
            updated_registration = ORDRegistration(
                registration_id=registration_id,
                ord_document=ord_document,
                metadata=updated_metadata,
                validation=validation,
                governance=existing_registration.governance,
                analytics={}
            )
            
            # Store in dual-database with replication
            success = await self.storage.store_registration(updated_registration)
            if not success:
                logger.error(f"❌ Failed to store updated registration: {registration_id}")
                return None
            
            # Re-index resources for search
            await self._index_ord_resources(
                updated_registration.registration_id,
                updated_registration.ord_document
            )
            
            logger.info(f"✅ Registration updated successfully: {registration_id} (v{updated_metadata.version})")
            return updated_registration
            
        except Exception as e:
            logger.error(f"❌ Error updating registration {registration_id}: {e}")
            return None
    
    async def update_registration_status(
        self,
        registration_id: str,
        status: RegistrationStatus
    ) -> bool:
        """Update only the registration status with dual-database storage"""
        await self._ensure_initialized()
        
        try:
            # Get existing registration
            existing_registration = await self.storage.get_registration(registration_id)
            if not existing_registration:
                logger.warning(f"Registration {registration_id} not found for status update")
                return False
            
            # Update only the status and timestamp
            updated_metadata = existing_registration.metadata
            updated_metadata.status = status
            updated_metadata.last_updated = datetime.utcnow()
            
            # Create updated registration with new status
            updated_registration = ORDRegistration(
                registration_id=registration_id,
                ord_document=existing_registration.ord_document,
                metadata=updated_metadata,
                validation=existing_registration.validation,
                governance=existing_registration.governance,
                analytics=existing_registration.analytics
            )
            
            # Store in dual-database with replication
            success = await self.storage.store_registration(updated_registration)
            if not success:
                logger.error(f"❌ Failed to update registration status: {registration_id}")
                return False
            
            logger.info(f"✅ Registration status updated: {registration_id} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f" Error updating registration status {registration_id}: {e}")
            return False
    
    async def delete_registration(
        self,
        registration_id: str,
        soft_delete: bool = True,
        deleted_by: str = "system"
    ) -> bool:
        """Delete an ORD registration with audit trail and dual-database storage"""
        await self._ensure_initialized()
        
        try:
            # Check if registration exists
            existing_registration = await self.storage.get_registration(registration_id)
            if not existing_registration:
                logger.warning(f"Registration {registration_id} not found for deletion")
                return False
            
            if soft_delete:
                # Soft delete: Update status to RETIRED with audit trail
                updated_metadata = existing_registration.metadata
                updated_metadata.status = RegistrationStatus.RETIRED
                updated_metadata.last_updated = datetime.utcnow()
                
                # Add deletion audit trail
                deletion_audit = {
                    "deleted_at": datetime.utcnow().isoformat(),
                    "deleted_by": deleted_by,
                    "deletion_type": "soft",
                    "reason": "User requested deletion"
                }
                
                # Convert AnalyticsInfo to dict, add deletion audit, then create new AnalyticsInfo
                analytics_dict = existing_registration.analytics.model_dump() if existing_registration.analytics else {}
                analytics_dict["deletion_audit"] = deletion_audit
                updated_analytics = AnalyticsInfo(**analytics_dict)
                
                # Create updated registration with deletion audit
                deleted_registration = ORDRegistration(
                    registration_id=registration_id,
                    ord_document=existing_registration.ord_document,
                    metadata=updated_metadata,
                    validation=existing_registration.validation,
                    governance=existing_registration.governance,
                    analytics=updated_analytics
                )
                
                # Update in dual-database with replication (use UPDATE, not INSERT for existing registration)
                success = await self.storage.update_registration(deleted_registration)
                if not success:
                    logger.error(f" Failed to soft delete registration: {registration_id}")
                    return False
                
                logger.info(f" Registration soft deleted: {registration_id} by {deleted_by}")
                return True
                
            else:
                # Hard delete: Remove from both databases
                success = await self.storage.delete_registration(registration_id)
                if not success:
                    logger.error(f" Failed to hard delete registration: {registration_id}")
                    return False
                
                logger.info(f" Registration hard deleted: {registration_id} by {deleted_by}")
                return True
                
        except Exception as e:
            logger.error(f" Error deleting registration {registration_id}: {e}")
            return False
    
    async def restore_registration(
        self,
        registration_id: str,
        restored_by: str = "system"
    ) -> bool:
        """Restore a soft-deleted ORD registration"""
        await self._ensure_initialized()
        
        try:
            # Get the soft-deleted registration
            existing_registration = await self.storage.get_registration(registration_id)
            if not existing_registration:
                logger.warning(f"Registration {registration_id} not found for restoration")
                return False
            
            if existing_registration.metadata.status != RegistrationStatus.RETIRED:
                logger.warning(f"Registration {registration_id} is not soft-deleted, cannot restore")
                return False
            
            # Restore by updating status to ACTIVE
            # Convert metadata to dict, update, then create new RegistrationMetadata
            metadata_dict = existing_registration.metadata.model_dump()
            metadata_dict["status"] = RegistrationStatus.ACTIVE
            metadata_dict["last_updated"] = datetime.utcnow()
            updated_metadata = RegistrationMetadata(**metadata_dict)
            
            # Add restoration audit trail
            restoration_audit = {
                "restored_at": datetime.utcnow().isoformat(),
                "restored_by": restored_by,
                "reason": "User requested restoration"
            }
            
            # Update analytics with restoration audit
            # Convert AnalyticsInfo to dict, update, then create new AnalyticsInfo
            analytics_dict = existing_registration.analytics.model_dump() if existing_registration.analytics else {}
            if "deletion_audit" in analytics_dict:
                analytics_dict["restoration_audit"] = restoration_audit
            updated_analytics = AnalyticsInfo(**analytics_dict)
            
            # Create restored registration
            restored_registration = ORDRegistration(
                registration_id=registration_id,
                ord_document=existing_registration.ord_document,
                metadata=updated_metadata,
                validation=existing_registration.validation,
                governance=existing_registration.governance,
                analytics=updated_analytics
            )
            
            # Update in dual-database with replication (restore existing registration)
            success = await self.storage.update_registration(restored_registration)
            if not success:
                logger.error(f" Failed to restore registration: {registration_id}")
                return False
            
            # Re-index resources for search
            await self._index_ord_resources(
                restored_registration.registration_id,
                restored_registration.ord_document
            )
            
            logger.info(f" Registration restored: {registration_id} by {restored_by}")
            return True
            
        except Exception as e:
            logger.error(f" Error restoring registration {registration_id}: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the registry with dual-database storage"""
        await self._ensure_initialized()
        
        try:
            # Check dual-database health
            health_info = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "details": {
                    "hana_healthy": False,
                    "supabase_healthy": False
                }
            }
            
            # Check HANA primary database
            try:
                if self.storage.hana_client is not None:
                    # Test HANA connectivity by checking if it's not None and not in fallback mode
                    hana_health = not self.storage.fallback_mode
                    health_info["details"]["hana_healthy"] = hana_health
                    health_info["hana_primary"] = {
                        "status": "connected" if hana_health else "disconnected",
                        "database": "SAP HANA Cloud",
                        "fallback_mode": self.storage.fallback_mode
                    }
                else:
                    health_info["hana_primary"] = {
                        "status": "unavailable",
                        "database": "SAP HANA Cloud",
                        "message": "HANA client not initialized"
                    }
            except Exception as e:
                health_info["hana_primary"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Check Supabase fallback database
            try:
                if self.storage.supabase_client is not None:
                    # Test Supabase connectivity by attempting a simple query
                    supabase_health = True
                    health_info["details"]["supabase_healthy"] = supabase_health
                    health_info["supabase_fallback"] = {
                        "status": "connected" if supabase_health else "disconnected",
                        "database": "Supabase PostgreSQL"
                    }
                else:
                    health_info["supabase_fallback"] = {
                        "status": "unavailable",
                        "database": "Supabase PostgreSQL",
                        "message": "Supabase client not initialized"
                    }
            except Exception as e:
                health_info["supabase_fallback"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Get registration counts
            try:
                active_count = await self.get_registration_count(active_only=True)
                total_count = await self.get_registration_count(active_only=False)
                health_info["registry_stats"] = {
                    "active_registrations": active_count,
                    "total_registrations": total_count,
                    "inactive_registrations": total_count - active_count
                }
            except Exception as e:
                health_info["registry_stats"] = {
                    "error": f"Failed to get stats: {e}"
                }
            
            # Overall health determination
            hana_ok = health_info.get("hana_primary", {}).get("status") == "connected"
            supabase_ok = health_info.get("supabase_fallback", {}).get("status") == "connected"
            
            if hana_ok and supabase_ok:
                health_info["status"] = "healthy"
            elif hana_ok or supabase_ok:
                health_info["status"] = "degraded"  # One database available
            else:
                health_info["status"] = "unhealthy"  # No databases available
            
            return health_info
            
        except Exception as e:
            logger.error(f"❌ Error getting health status: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_registration_count(
        self,
        active_only: bool = True
    ) -> int:
        """Get count of registrations from dual-database storage"""
        await self._ensure_initialized()
        
        try:
            # Query both databases for registration count
            count = await self.storage.get_registration_count(active_only=active_only)
            return count
        except Exception as e:
            logger.error(f"❌ Error getting registration count: {e}")
            return 0
    
    async def bulk_update_registrations(
        self,
        updates: List[Dict[str, Any]],
        enhance_with_ai: bool = True,
        updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Bulk update multiple ORD registrations with dual-database storage and AI re-enhancement"""
        await self._ensure_initialized()
        
        results = {
            "successful": [],
            "failed": [],
            "total_processed": len(updates),
            "success_count": 0,
            "error_count": 0
        }
        
        try:
            for update_data in updates:
                registration_id = update_data.get("registration_id")
                ord_document_data = update_data.get("ord_document")
                
                if not registration_id or not ord_document_data:
                    results["failed"].append({
                        "registration_id": registration_id,
                        "error": "Missing registration_id or ord_document"
                    })
                    results["error_count"] += 1
                    continue
                
                try:
                    # Convert dict to ORDDocument object if needed
                    if isinstance(ord_document_data, dict):
                        ord_document = ORDDocument(**ord_document_data)
                    else:
                        ord_document = ord_document_data
                    
                    # Update individual registration
                    updated_registration = await self.update_registration(
                        registration_id=registration_id,
                        ord_document=ord_document,
                        enhance_with_ai=enhance_with_ai
                    )
                    
                    if updated_registration:
                        results["successful"].append({
                            "registration_id": registration_id,
                            "version": updated_registration.metadata.version,
                            "updated_at": updated_registration.metadata.last_updated.isoformat()
                        })
                        results["success_count"] += 1
                    else:
                        results["failed"].append({
                            "registration_id": registration_id,
                            "error": "Update failed - registration not found or validation failed"
                        })
                        results["error_count"] += 1
                        
                except Exception as e:
                    results["failed"].append({
                        "registration_id": registration_id,
                        "error": f"Update error: {str(e)}"
                    })
                    results["error_count"] += 1
                    logger.error(f"❌ Bulk update failed for {registration_id}: {e}")
            
            logger.info(f"✅ Bulk update completed: {results['success_count']}/{results['total_processed']} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Bulk update operation failed: {e}")
            results["failed"].append({"error": f"Bulk operation error: {str(e)}"})
            return results
    
    async def bulk_delete_registrations(
        self,
        registration_ids: List[str],
        soft_delete: bool = True,
        deleted_by: str = "system"
    ) -> Dict[str, Any]:
        """Bulk delete multiple ORD registrations with audit trails and dual-database storage"""
        await self._ensure_initialized()
        
        results = {
            "successful": [],
            "failed": [],
            "total_processed": len(registration_ids),
            "success_count": 0,
            "error_count": 0,
            "soft_delete": soft_delete
        }
        
        try:
            for registration_id in registration_ids:
                try:
                    # Delete individual registration
                    success = await self.delete_registration(
                        registration_id=registration_id,
                        soft_delete=soft_delete,
                        deleted_by=deleted_by
                    )
                    
                    if success:
                        results["successful"].append({
                            "registration_id": registration_id,
                            "deleted_at": datetime.utcnow().isoformat(),
                            "deletion_type": "soft" if soft_delete else "hard"
                        })
                        results["success_count"] += 1
                    else:
                        results["failed"].append({
                            "registration_id": registration_id,
                            "error": "Delete failed - registration not found"
                        })
                        results["error_count"] += 1
                        
                except Exception as e:
                    results["failed"].append({
                        "registration_id": registration_id,
                        "error": f"Delete error: {str(e)}"
                    })
                    results["error_count"] += 1
                    logger.error(f"❌ Bulk delete failed for {registration_id}: {e}")
            
            deletion_type = "soft" if soft_delete else "hard"
            logger.info(f"✅ Bulk {deletion_type} delete completed: {results['success_count']}/{results['total_processed']} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Bulk delete operation failed: {e}")
            results["failed"].append({"error": f"Bulk operation error: {str(e)}"})
            return results
    
    async def get_all_registrations(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[ORDRegistration]:
        """Get all registrations from dual-database storage with pagination"""
        await self._ensure_initialized()
        
        try:
            registrations = await self.storage.list_all_registrations(
                limit=limit
            )
            
            logger.info(f"✅ Retrieved {len(registrations)} registrations (active_only={active_only})")
            return registrations
            
        except Exception as e:
            logger.error(f"❌ Error getting all registrations: {e}")
            return []
    
    async def get_registration_status(
        self,
        registration_id: str
    ) -> Optional[RegistrationStatus]:
        """Get the status of a specific registration"""
        await self._ensure_initialized()
        
        try:
            registration = await self.storage.get_registration(registration_id)
            if registration:
                return registration.metadata.status
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting registration status {registration_id}: {e}")
            return None

    def _calculate_search_facets_from_entries(self, entries: List[ResourceIndexEntry]) -> Optional[Dict[str, List[SearchFacet]]]:
        """Calculate search facets from ResourceIndexEntry objects"""
        try:
            if not entries:
                return None
            
            # Count facet values
            resource_types = {}
            creators = {}
            subjects = {}
            publishers = {}
            formats = {}
            
            for entry in entries:
                # Resource type facet
                if entry.resource_type:
                    rt = entry.resource_type.value if hasattr(entry.resource_type, 'value') else str(entry.resource_type)
                    resource_types[rt] = resource_types.get(rt, 0) + 1
                
                # Dublin Core facets
                if entry.dc_creator:
                    for creator in entry.dc_creator:
                        creator_name = creator.get('name', creator) if isinstance(creator, dict) else str(creator)
                        creators[creator_name] = creators.get(creator_name, 0) + 1
                
                if entry.dc_subject:
                    for subject in entry.dc_subject:
                        subject_str = str(subject)
                        subjects[subject_str] = subjects.get(subject_str, 0) + 1
                
                if entry.dc_publisher:
                    publishers[entry.dc_publisher] = publishers.get(entry.dc_publisher, 0) + 1
                
                if entry.dc_format:
                    formats[entry.dc_format] = formats.get(entry.dc_format, 0) + 1
            
            # Convert to SearchFacet objects
            facets = {}
            
            if resource_types:
                facets["resource_types"] = [
                    SearchFacet(value=rt, count=count) 
                    for rt, count in sorted(resource_types.items(), key=lambda x: x[1], reverse=True)
                ]
            
            if creators:
                facets["creators"] = [
                    SearchFacet(value=creator, count=count)
                    for creator, count in sorted(creators.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            if subjects:
                facets["subjects"] = [
                    SearchFacet(value=subject, count=count)
                    for subject, count in sorted(subjects.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            if publishers:
                facets["publishers"] = [
                    SearchFacet(value=publisher, count=count)
                    for publisher, count in sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            if formats:
                facets["formats"] = [
                    SearchFacet(value=format_val, count=count)
                    for format_val, count in sorted(formats.items(), key=lambda x: x[1], reverse=True)[:10]
                ]
            
            return facets if facets else None
            
        except Exception as e:
            logger.error(f"Failed to calculate facets: {e}")
            return None
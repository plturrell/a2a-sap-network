import os
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime

from .models import (
    ORDDocument, RegistrationRequest, RegistrationResponse,
    SearchRequest, SearchResult, ResourceIndexEntry,
    DublinCoreValidationRequest, DublinCoreValidationResponse
)
from .service import ORDRegistryService

router = APIRouter(prefix="/api/v1/ord", tags=["ORD Registry"])

# Initialize ORD Registry Service with dual-database and AI features
ord_registry = ORDRegistryService(base_url=os.getenv("ORD_REGISTRY_URL", "http://localhost:8091") + "/api/v1/ord")

# Dependency to ensure service is initialized
async def ensure_ord_registry_initialized():
    """Dependency to ensure ORD Registry Service is initialized"""
    if not ord_registry.initialized:
        await ord_registry.initialize()
    return ord_registry


@router.post("/register", response_model=RegistrationResponse)
async def register_ord_document(
    registration_request: RegistrationRequest,
    request: Request,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Register a new ORD document in the registry"""
    try:
        result = await registry.register_ord_document(
            ord_document=registration_request.ord_document,
            registered_by=registration_request.registered_by,
            tags=registration_request.tags,
            labels=registration_request.labels
        )
        
        if not result or not result.registration_id:
            # Get the actual validation errors from the service logs
            error_message = "Registration failed"
            
            # Check if we can get more specific error information
            try:
                # Try to validate the document to get specific errors
                validation_result = await registry._validate_ord_document(registration_request.ord_document)
                if validation_result.errors:
                    error_message = f"Validation failed: {', '.join(validation_result.errors)}"
                else:
                    # Validation passed but storage failed
                    error_message = "Registration failed: Storage error - check database connectivity"
                    
                    # Check storage status
                    if registry.storage:
                        if registry.storage.fallback_mode:
                            error_message += " (Running in fallback mode - HANA unavailable)"
                        if not registry.storage.hana_client and not registry.storage.sqlite_client:
                            error_message += " (No database connections available)"
            except Exception as e:
                error_message = f"Registration failed: {str(e)}"
            
            return JSONResponse(
                status_code=400,
                content={"errors": [error_message]}
            )
        
        # Build response data from ORDRegistration object
        response_data = {
            "registration_id": result.registration_id,
            "status": result.metadata.status.value,
            "validation_results": result.validation,
            "registered_at": result.metadata.registered_at,
            "registry_url": f"{request.url.scheme}://{request.url.netloc}/api/v1/ord"
        }
        
        return RegistrationResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/register/{registration_id}")
async def get_registration(registration_id: str):
    """Get registration details by ID"""
    registration = await ord_registry.get_registration(registration_id)
    
    if not registration:
        raise HTTPException(
            status_code=404,
            detail=f"Registration {registration_id} not found"
        )
    
    return registration.dict()


@router.put("/register/{registration_id}")
async def update_ord_document(
    registration_id: str, 
    ord_document: ORDDocument,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Update an existing ORD document"""
    try:
        # Update the ORD document
        updated_registration = await registry.update_registration(
            registration_id=registration_id,
            ord_document=ord_document,
            enhance_with_ai=True
        )
        
        if not updated_registration:
            raise HTTPException(
                status_code=404,
                detail=f"Registration {registration_id} not found or update failed"
            )
        
        return {
            "registration_id": registration_id,
            "status": updated_registration.metadata.status.value,
            "version": updated_registration.metadata.version,
            "updated_at": updated_registration.metadata.last_updated
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/register/{registration_id}")
async def delete_registration(
    registration_id: str,
    soft_delete: bool = True,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Delete a registration (soft delete by default)"""
    try:
        # Perform soft delete by default
        success = await registry.delete_registration(
            registration_id=registration_id,
            soft_delete=soft_delete,
            deleted_by="api_user"  # In production, get from auth context
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Registration {registration_id} not found"
            )
        
        return {
            "message": f"Registration {registration_id} {'soft' if soft_delete else 'hard'} deleted successfully",
            "registration_id": registration_id,
            "deleted_at": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResult)
async def search_resources(search_request: SearchRequest):
    """Search for resources in the registry"""
    try:
        results = await ord_registry.search_resources(search_request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/{ord_id}/analytics")
async def get_resource_analytics(
    ord_id: str,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Get analytics for a specific resource"""
    try:
        if registry.enhanced_search:
            analytics = await registry.enhanced_search.get_resource_analytics(ord_id)
            return {"analytics": analytics}
        else:
            return {"analytics": None, "message": "Enhanced search not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/{ord_id}")
async def get_resource_by_ord_id(ord_id: str):
    """Get a specific resource by ORD ID"""
    resource = await ord_registry.get_resource_by_ord_id(ord_id)
    
    if not resource:
        raise HTTPException(
            status_code=404,
            detail=f"Resource with ORD ID {ord_id} not found"
        )
    
    return resource.dict()


@router.get("/browse")
async def browse_resources(
    category: Optional[str] = None,
    domain: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """Browse resources by category or domain"""
    search_request = SearchRequest(
        category=category,
        domain=domain,
        page=page,
        page_size=page_size
    )
    
    try:
        results = await ord_registry.search_resources(search_request)
        return results.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/register/{registration_id}/status")
async def get_registration_status(registration_id: str):
    """Get registration status"""
    registration = await ord_registry.get_registration(registration_id)
    
    if not registration:
        raise HTTPException(
            status_code=404,
            detail=f"Registration {registration_id} not found"
        )
    
    return {
        "registration_id": registration_id,
        "status": registration.metadata.status,
        "last_updated": registration.metadata.last_updated,
        "validation": registration.validation.dict()
    }


@router.get("/register/{registration_id}/validation")
async def get_validation_report(registration_id: str):
    """Get detailed validation report"""
    registration = await ord_registry.get_registration(registration_id)
    
    if not registration:
        raise HTTPException(
            status_code=404,
            detail=f"Registration {registration_id} not found"
        )
    
    return {
        "registration_id": registration_id,
        "validation_results": registration.validation.dict(),
        "compliance_score": registration.validation.compliance_score,
        "validated_at": registration.metadata.registered_at
    }


@router.get("/analytics/{registration_id}")
async def get_usage_analytics(registration_id: str):
    """Get usage analytics for a registration"""
    registration = await ord_registry.get_registration(registration_id)
    
    if not registration:
        raise HTTPException(
            status_code=404,
            detail=f"Registration {registration_id} not found"
        )
    
    return {
        "registration_id": registration_id,
        "analytics": registration.analytics.dict()
    }


@router.post("/dublincore/validate", response_model=DublinCoreValidationResponse)
async def validate_dublin_core(request: DublinCoreValidationRequest):
    """Validate Dublin Core metadata for compliance and quality"""
    try:
        validation_result = await ord_registry.validate_dublin_core_metadata(request)
        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint with Dublin Core metrics"""
    try:
        health_status = await ord_registry.get_health_status()
        return health_status
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/metrics")
async def get_metrics():
    """Get registry metrics including Dublin Core quality metrics"""
    health = await ord_registry.get_health_status()
    
    return {
        "ord_registry_total_registrations": len(ord_registry.registrations),
        "ord_registry_active_resources": len(ord_registry.resource_index),
        "ord_registry_dublin_core_enabled": health["metrics"]["dublin_core_enabled"],
        "ord_registry_quality_score": health["metrics"]["average_quality_score"],
        "ord_registry_iso15836_compliance": health["standards_compliance"]["iso15836_compliance_rate"],
        "ord_registry_rfc5013_compliance": health["standards_compliance"]["rfc5013_compliance_rate"],
        "ord_registry_up": 1
    }


@router.get("/blockchain/status")
async def get_blockchain_status(
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Get blockchain integration status"""
    try:
        status = await registry.get_blockchain_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/register/{registration_id}/verify")
async def verify_document_integrity(
    registration_id: str,
    version: str = None,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Verify document integrity using blockchain records"""
    try:
        verification = await registry.verify_document_integrity(registration_id, version)
        
        if verification.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Registration not found")
        elif verification.get("status") == "unavailable":
            raise HTTPException(status_code=503, detail="Blockchain verification not available")
        
        return verification
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/register/{registration_id}/blockchain-history")
async def get_document_blockchain_history(
    registration_id: str,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Get complete blockchain history for a document"""
    try:
        history = await registry.get_document_blockchain_history(registration_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register/{registration_id}/blockchain-audit")
async def create_blockchain_audit_entry(
    registration_id: str,
    audit_data: dict,
    registry: ORDRegistryService = Depends(ensure_ord_registry_initialized)
):
    """Create a blockchain audit trail entry"""
    try:
        if not registry.blockchain_integration:
            raise HTTPException(status_code=503, detail="Blockchain integration not available")
        
        audit_entry = await registry.blockchain_integration.create_audit_trail(
            registration_id=registration_id,
            operation=audit_data.get("operation", "manual_audit"),
            user=audit_data.get("user", "api_user"),
            details=audit_data.get("details", {})
        )
        
        return {
            "message": "Audit entry created successfully",
            "audit_entry": audit_entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
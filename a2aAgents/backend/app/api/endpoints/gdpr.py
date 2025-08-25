"""
GDPR Compliance API Endpoints
Implements GDPR-specific data protection features
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



from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
import logging

from ...core.gdprCompliance import (
    get_gdpr_manager,
    LawfulBasis,
    DataSubjectRight,
    ProcessingPurpose
)
from ..deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class ConsentRequest(BaseModel):
    """Request model for recording consent"""
    purpose: ProcessingPurpose
    description: str = Field(..., min_length=10)
    data_categories: List[str] = Field(..., min_items=1)
    duration_days: Optional[int] = Field(None, ge=1, le=3650)


class ConsentWithdrawalRequest(BaseModel):
    """Request model for withdrawing consent"""
    consent_id: str
    reason: Optional[str] = None


class DataSubjectRequestCreate(BaseModel):
    """Request model for creating data subject request"""
    request_type: DataSubjectRight
    verification_email: Optional[EmailStr] = None


class ProcessingActivityRequest(BaseModel):
    """Request model for registering processing activity"""
    activity_name: str
    purpose: ProcessingPurpose
    lawful_basis: LawfulBasis
    data_categories: List[str]
    retention_period_days: int = Field(..., ge=1)
    recipients: List[str]
    description: str


@router.post("/consent")
async def record_consent(
    request: ConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Record user consent for data processing

    - Creates immutable consent record
    - Supports time-limited consent
    - Full audit trail maintained
    """
    try:
        gdpr_manager = get_gdpr_manager()

        # Get request context
        ip_address = current_user.get("ip_address")
        user_agent = current_user.get("user_agent")

        consent_id = await gdpr_manager.record_consent(
            user_id=current_user["user_id"],
            purpose=request.purpose,
            description=request.description,
            data_categories=request.data_categories,
            duration_days=request.duration_days,
            ip_address=ip_address,
            user_agent=user_agent
        )

        return {
            "status": "success",
            "consent_id": consent_id,
            "message": "Consent recorded successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to record consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/consent")
async def withdraw_consent(
    request: ConsentWithdrawalRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Withdraw previously given consent

    - Immediate effect on data processing
    - Maintains withdrawal record
    - Cannot be reversed
    """
    try:
        gdpr_manager = get_gdpr_manager()

        success = await gdpr_manager.withdraw_consent(
            user_id=current_user["user_id"],
            consent_id=request.consent_id,
            reason=request.reason
        )

        if not success:
            raise HTTPException(status_code=404, detail="Consent not found or already withdrawn")

        return {
            "status": "success",
            "message": "Consent withdrawn successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to withdraw consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consent")
async def list_consents(
    active_only: bool = Query(True, description="Show only active consents"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List user's consent records

    - Shows all or only active consents
    - Includes consent details and status
    """
    try:
        gdpr_manager = get_gdpr_manager()

        if active_only:
            consents = await gdpr_manager.get_active_consents(current_user["user_id"])
        else:
            consents = await gdpr_manager.get_consent_history(current_user["user_id"])

        # Convert to response format
        consent_list = []
        for consent in consents:
            consent_list.append({
                "consent_id": consent.consent_id,
                "purpose": consent.purpose.value,
                "description": consent.description,
                "data_categories": consent.data_categories,
                "granted_at": consent.granted_at.isoformat(),
                "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
                "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                "withdrawal_reason": consent.withdrawal_reason,
                "is_active": consent.withdrawn_at is None and (
                    consent.expires_at is None or consent.expires_at > datetime.utcnow()
                )
            })

        return {
            "count": len(consent_list),
            "consents": consent_list
        }

    except Exception as e:
        logger.error(f"Failed to list consents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consent/check")
async def check_consent(
    purpose: ProcessingPurpose = Query(..., description="Processing purpose to check"),
    data_categories: Optional[List[str]] = Query(None, description="Specific data categories"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if user has given consent for specific processing

    - Validates against active consents
    - Checks data category coverage
    """
    try:
        gdpr_manager = get_gdpr_manager()

        has_consent, consent_record = await gdpr_manager.check_consent(
            user_id=current_user["user_id"],
            purpose=purpose,
            data_categories=data_categories
        )

        response = {
            "has_consent": has_consent,
            "purpose": purpose.value,
            "checked_at": datetime.utcnow().isoformat()
        }

        if has_consent and consent_record:
            response["consent_id"] = consent_record.consent_id
            response["granted_at"] = consent_record.granted_at.isoformat()
            response["expires_at"] = consent_record.expires_at.isoformat() if consent_record.expires_at else None

        return response

    except Exception as e:
        logger.error(f"Failed to check consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-subject-request")
async def create_data_subject_request(
    request: DataSubjectRequestCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a GDPR data subject request

    - Supports all GDPR rights (access, erasure, portability, etc.)
    - Initiates verification process
    - 30-day response commitment
    """
    try:
        gdpr_manager = get_gdpr_manager()

        request_id = await gdpr_manager.create_subject_request(
            user_id=current_user["user_id"],
            request_type=request.request_type,
            verification_method="email" if request.verification_email else "token"
        )

        # Generate verification token
        import hashlib
        verification_token = hashlib.sha256(
            f"{request_id}:{current_user['user_id']}".encode()
        ).hexdigest()[:8]

        return {
            "status": "success",
            "request_id": request_id,
            "request_type": request.request_type.value,
            "verification_required": True,
            "verification_token": verification_token,
            "message": f"Request created. Please verify using token: {verification_token}"
        }

    except Exception as e:
        logger.error(f"Failed to create data subject request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-subject-request/{request_id}/verify")
async def verify_data_subject_request(
    request_id: str,
    verification_token: str = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify a data subject request

    - Required before processing
    - Ensures request authenticity
    """
    try:
        gdpr_manager = get_gdpr_manager()

        verified = await gdpr_manager.verify_request(request_id, verification_token)

        if not verified:
            raise HTTPException(status_code=400, detail="Invalid verification token")

        return {
            "status": "success",
            "message": "Request verified successfully",
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-subject-request/{request_id}/process")
async def process_data_subject_request(
    request_id: str,
    confirm: bool = Query(False, description="Confirm irreversible actions"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Process a verified data subject request

    - Executes the requested action
    - Returns results based on request type
    """
    try:
        gdpr_manager = get_gdpr_manager()

        # Get request details
        request = gdpr_manager.subject_# WARNING: requests.get usage violates A2A protocol - must use blockchain messaging
        # requests\.get(request_id)
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Verify ownership
        if request.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Process based on type
        if request.request_type == DataSubjectRight.ACCESS:
            result = await gdpr_manager.process_access_request(request_id)

        elif request.request_type == DataSubjectRight.ERASURE:
            result = await gdpr_manager.process_erasure_request(request_id, confirm)

        elif request.request_type == DataSubjectRight.DATA_PORTABILITY:
            result = await gdpr_manager.process_portability_request(request_id)

        else:
            raise HTTPException(status_code=400, detail=f"Request type {request.request_type} not implemented")

        return {
            "status": "success",
            "request_id": request_id,
            "request_type": request.request_type.value,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-subject-request")
async def list_data_subject_requests(
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List user's data subject requests

    - Shows request history
    - Includes status and results
    """
    try:
        gdpr_manager = get_gdpr_manager()

        # Get user's requests
        user_requests = [
            r for r in gdpr_manager.subject_requests.values()
            if r.user_id == current_user["user_id"]
        ]

        # Filter by status if specified
        if status:
            user_requests = [r for r in user_requests if r.status == status]

        # Convert to response format
        request_list = []
        for req in user_requests:
            request_list.append({
                "request_id": req.request_id,
                "request_type": req.request_type.value,
                "status": req.status,
                "submitted_at": req.submitted_at.isoformat(),
                "completed_at": req.completed_at.isoformat() if req.completed_at else None,
                "verified": req.verified
            })

        return {
            "count": len(request_list),
            "requests": request_list
        }

    except Exception as e:
        logger.error(f"Failed to list requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing-activity")
async def register_processing_activity(
    request: ProcessingActivityRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Register a data processing activity (Article 30)

    - Required for GDPR compliance
    - Documents processing purposes and legal basis
    - Admin only
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")

        gdpr_manager = get_gdpr_manager()

        activity_id = await gdpr_manager.register_processing_activity(
            activity_name=request.activity_name,
            purpose=request.purpose,
            lawful_basis=request.lawful_basis,
            data_categories=request.data_categories,
            retention_period_days=request.retention_period_days,
            recipients=request.recipients,
            description=request.description
        )

        return {
            "status": "success",
            "activity_id": activity_id,
            "message": "Processing activity registered successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register processing activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing-activities")
async def list_processing_activities(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List registered processing activities

    - Shows all data processing purposes
    - Includes legal basis and retention
    """
    try:
        gdpr_manager = get_gdpr_manager()

        activities = list(gdpr_manager.processing_registry.values())

        return {
            "count": len(activities),
            "activities": activities
        }

    except Exception as e:
        logger.error(f"Failed to list processing activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-minimization-check")
async def check_data_minimization(
    purpose: ProcessingPurpose = Query(..., description="Processing purpose"),
    requested_fields: List[str] = Body(..., description="Fields requested for processing"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if data request adheres to minimization principle

    - Validates against purpose requirements
    - Identifies excessive data collection
    """
    try:
        gdpr_manager = get_gdpr_manager()

        compliant, excessive_fields = await gdpr_manager.check_data_minimization(
            purpose=purpose,
            requested_fields=requested_fields
        )

        return {
            "compliant": compliant,
            "purpose": purpose.value,
            "requested_fields": requested_fields,
            "excessive_fields": excessive_fields,
            "recommendation": "Remove excessive fields" if excessive_fields else "Data request is minimal"
        }

    except Exception as e:
        logger.error(f"Failed to check data minimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/privacy-rights")
async def get_privacy_rights(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get information about GDPR privacy rights

    - Educational endpoint
    - Lists all available rights
    """
    rights = {
        DataSubjectRight.ACCESS.value: {
            "name": "Right of Access",
            "article": "Article 15",
            "description": "Obtain confirmation and copy of your personal data being processed"
        },
        DataSubjectRight.RECTIFICATION.value: {
            "name": "Right to Rectification",
            "article": "Article 16",
            "description": "Correct inaccurate personal data"
        },
        DataSubjectRight.ERASURE.value: {
            "name": "Right to Erasure (Right to be Forgotten)",
            "article": "Article 17",
            "description": "Request deletion of your personal data"
        },
        DataSubjectRight.RESTRICT_PROCESSING.value: {
            "name": "Right to Restrict Processing",
            "article": "Article 18",
            "description": "Limit how your personal data is used"
        },
        DataSubjectRight.DATA_PORTABILITY.value: {
            "name": "Right to Data Portability",
            "article": "Article 20",
            "description": "Receive your data in a portable format"
        },
        DataSubjectRight.OBJECT.value: {
            "name": "Right to Object",
            "article": "Article 21",
            "description": "Object to processing of your personal data"
        }
    }

    return {
        "rights": rights,
        "info": "You can exercise any of these rights through the data subject request endpoint"
    }


# Export router
__all__ = ["router"]
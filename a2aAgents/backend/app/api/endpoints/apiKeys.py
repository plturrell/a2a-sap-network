"""
API Key Management Endpoints
Manage API keys for request signing
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import logging

from ...core.requestSigning import get_signing_service
from ...core.securityMonitoring import report_security_event, EventType, ThreatLevel
from ..deps import get_current_user
from ...core.rbac import Permission

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API key"""
    key_id: str = Field(..., description="Unique identifier for the API key")
    permissions: List[str] = Field(..., description="List of permissions for the key")
    description: Optional[str] = Field(None, description="Description of the key's purpose")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Days until key expires (1-365, None for no expiration)")


class APIKeyResponse(BaseModel):
    """Response model for API key operations"""
    key_id: str
    secret: Optional[str] = None  # Only returned on creation
    active: bool
    permissions: List[str]
    created_at: Optional[str] = None
    rotated_at: Optional[str] = None
    revoked_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_expired: Optional[bool] = None


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user=Depends(get_current_user)
) -> APIKeyResponse:
    """
    Create a new API key for request signing

    - **key_id**: Unique identifier for the API key
    - **permissions**: List of permissions (read, write, admin, a2a)
    - **description**: Optional description of the key's purpose

    Returns the key ID and secret. The secret is only shown once!
    """
    try:
        # Check user permissions
        if not current_user.has_permission(Permission.ADMIN):
            raise HTTPException(status_code=403, detail="Admin permission required")

        signing_service = get_signing_service()

        # Validate permissions
        valid_permissions = {"read", "write", "admin", "a2a"}
        invalid_perms = set(request.permissions) - valid_permissions
        if invalid_perms:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid permissions: {', '.join(invalid_perms)}"
            )

        # Calculate expiration date if specified
        expires_at = None
        if request.expires_in_days:
            from datetime import datetime, timedelta
            expires_at = (datetime.utcnow() + timedelta(days=request.expires_in_days)).isoformat()

        # Create API key
        result = signing_service.create_api_key(
            key_id=request.key_id,
            permissions=request.permissions,
            expires_at=expires_at
        )

        # Log security event
        await report_security_event(
            EventType.ACCESS_DENIED,  # Using as audit event
            ThreatLevel.INFO,
            f"API key created: {request.key_id}",
            user_id=current_user.id,
            details={
                "key_id": request.key_id,
                "permissions": request.permissions,
                "created_by": current_user.id
            }
        )

        logger.info(f"API key created: {request.key_id} by user {current_user.id}")

        # Get key info
        key_info = signing_service.api_keys[request.key_id]

        return APIKeyResponse(
            key_id=result["key_id"],
            secret=result["secret"],  # Only returned on creation
            active=key_info["active"],
            permissions=key_info["permissions"],
            created_at=key_info.get("created_at"),
            expires_at=key_info.get("expires_at"),
            is_expired=False  # New keys are never expired
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user=Depends(get_current_user)
) -> List[APIKeyResponse]:
    """
    List all API keys

    Returns list of API keys without secrets.
    """
    try:
        # Check user permissions
        if not current_user.has_permission(Permission.ADMIN):
            raise HTTPException(status_code=403, detail="Admin permission required")

        signing_service = get_signing_service()

        from datetime import datetime

        keys = []
        for key_id, key_info in signing_service.api_keys.items():
            # Check if key is expired
            is_expired = False
            expires_at = key_info.get("expires_at")
            if expires_at:
                is_expired = datetime.fromisoformat(expires_at.replace('Z', '+00:00')) < datetime.utcnow()

            keys.append(APIKeyResponse(
                key_id=key_id,
                active=key_info["active"],
                permissions=key_info["permissions"],
                created_at=key_info.get("created_at"),
                rotated_at=key_info.get("rotated_at"),
                revoked_at=key_info.get("revoked_at"),
                expires_at=expires_at,
                is_expired=is_expired
            ))

        return keys

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(
    key_id: str,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Rotate an API key (generate new secret)

    Returns the new secret. The secret is only shown once!
    """
    try:
        # Check user permissions
        if not current_user.has_permission(Permission.ADMIN):
            raise HTTPException(status_code=403, detail="Admin permission required")

        signing_service = get_signing_service()

        # Rotate key
        new_secret = signing_service.rotate_api_key(key_id)

        # Log security event
        await report_security_event(
            EventType.ACCESS_DENIED,  # Using as audit event
            ThreatLevel.INFO,
            f"API key rotated: {key_id}",
            user_id=current_user.id,
            details={
                "key_id": key_id,
                "rotated_by": current_user.id
            }
        )

        logger.info(f"API key rotated: {key_id} by user {current_user.id}")

        return {
            "key_id": key_id,
            "secret": new_secret,
            "message": "API key rotated successfully. Save the new secret - it won't be shown again!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rotate API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Revoke an API key

    The key will be immediately invalidated.
    """
    try:
        # Check user permissions
        if not current_user.has_permission(Permission.ADMIN):
            raise HTTPException(status_code=403, detail="Admin permission required")

        signing_service = get_signing_service()

        # Revoke key
        signing_service.revoke_api_key(key_id)

        # Log security event
        await report_security_event(
            EventType.ACCESS_DENIED,  # Using as audit event
            ThreatLevel.MEDIUM,
            f"API key revoked: {key_id}",
            user_id=current_user.id,
            details={
                "key_id": key_id,
                "revoked_by": current_user.id
            }
        )

        logger.info(f"API key revoked: {key_id} by user {current_user.id}")

        return {
            "key_id": key_id,
            "status": "revoked",
            "message": "API key has been revoked and is no longer valid"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api-keys/{key_id}")
async def get_api_key(
    key_id: str,
    current_user=Depends(get_current_user)
) -> APIKeyResponse:
    """
    Get details of a specific API key

    Returns key information without the secret.
    """
    try:
        # Check user permissions
        if not current_user.has_permission(Permission.ADMIN):
            raise HTTPException(status_code=403, detail="Admin permission required")

        signing_service = get_signing_service()

        if key_id not in signing_service.api_keys:
            raise HTTPException(status_code=404, detail="API key not found")

        key_info = signing_service.api_keys[key_id]

        # Check if key is expired
        from datetime import datetime
        is_expired = False
        expires_at = key_info.get("expires_at")
        if expires_at:
            is_expired = datetime.fromisoformat(expires_at.replace('Z', '+00:00')) < datetime.utcnow()

        return APIKeyResponse(
            key_id=key_id,
            active=key_info["active"],
            permissions=key_info["permissions"],
            created_at=key_info.get("created_at"),
            rotated_at=key_info.get("rotated_at"),
            revoked_at=key_info.get("revoked_at"),
            expires_at=expires_at,
            is_expired=is_expired
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]

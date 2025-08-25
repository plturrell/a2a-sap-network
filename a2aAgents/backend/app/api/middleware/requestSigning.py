"""
Request Signing Middleware
Automatically verifies signed API requests
"""
import http

import logging
from typing import Optional, List
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED

from ...core.requestSigning import get_signing_service
from ...core.securityMonitoring import report_security_event, EventType, ThreatLevel

logger = logging.getLogger(__name__)


class RequestSigningMiddleware(BaseHTTPMiddleware):
    """Middleware to verify signed API requests"""

    def __init__(self, app, excluded_paths: Optional[List[str]] = None, enforce_signing: bool = False):
        super().__init__(app)
        self.signing_service = get_signing_service()
        self.enforce_signing = enforce_signing

        # Paths that don't require signing
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
            "/api/v1/auth/login",  # Allow login without signing
            "/api/v1/auth/register",  # Allow registration without signing
        ]

        logger.info(f"Request Signing Middleware initialized (enforce={enforce_signing})")

    async def dispatch(self, request: Request, call_next):
        """Process request and verify signature if present"""

        # Check if path is excluded
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Check if request has signing headers
        has_signing_headers = all([
            request.headers.get("x-api-key-id"),
            request.headers.get("x-timestamp"),
            request.headers.get("x-nonce"),
            request.headers.get("x-signature")
        ])

        if has_signing_headers:
            # Verify signed request
            try:
                # Read body if present (for integrity check)
                body = None
                if request.method in ["POST", "PUT", "PATCH"]:
                    body = await request.body()
                    # Need to reconstruct request with body
                    from starlette.datastructures import Headers
                    from starlette.requests import Request as StarletteRequest

                    # Create new request with body available for downstream
                    request = StarletteRequest(
                        scope=request.scope,
                        receive=self._create_receive(body)
                    )

                # Extract query parameters
                query_params = dict(request.query_params.multi_items()) if request.query_params else None

                # Verify request
                is_valid, error_message = await self.signing_service.verify_request(
                    method=request.method,
                    path=request.url.path,
                    headers=dict(request.headers),
                    body=body,
                    query_params=query_params
                )

                if not is_valid:
                    logger.warning(f"Invalid signed request: {error_message}")
                    raise HTTPException(
                        status_code=HTTP_401_UNAUTHORIZED,
                        detail=f"Invalid request signature: {error_message}"
                    )

                # Add API key ID to request state for downstream use
                request.state.api_key_id = request.headers.get("x-api-key-id")
                request.state.is_signed_request = True

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error verifying signed request: {e}")
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Request signature verification failed"
                )

        elif self.enforce_signing:
            # Signing is required but headers are missing
            await report_security_event(
                EventType.UNAUTHORIZED_API_ACCESS,
                ThreatLevel.MEDIUM,
                f"Unsigned request to protected endpoint: {request.url.path}",
                source_ip=request.client.host if request.client else None,
                details={
                    "path": request.url.path,
                    "method": request.method
                }
            )

            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Request signing is required for this endpoint"
            )

        else:
            # Signing is optional and not present
            request.state.is_signed_request = False

        # Process request
        response = await call_next(request)

        # Add signature verification status header
        if hasattr(request.state, "is_signed_request"):
            response.headers["X-Signature-Verified"] = str(request.state.is_signed_request).lower()

        return response

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from signing requirements"""
        # Exact match
        if path in self.excluded_paths:
            return True

        # Prefix match for documentation
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True

        return False

    def _create_receive(self, body: bytes):
        """Create receive callable for request with body"""
        async def receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False
            }
        return receive


class APIKeyPermissionMiddleware(BaseHTTPMiddleware):
    """Middleware to check API key permissions for signed requests"""

    def __init__(self, app):
        super().__init__(app)
        self.signing_service = get_signing_service()

        # Map paths to required permissions
        self.permission_map = {
            "/api/v1/admin/": "admin",
            "/api/v1/data/": "write",
            "/api/v1/users/": "write",
            "/a2a/": "a2a",
        }

    async def dispatch(self, request: Request, call_next):
        """Check API key permissions for signed requests"""

        # Only check permissions for signed requests
        if not hasattr(request.state, "is_signed_request") or not request.state.is_signed_request:
            return await call_next(request)

        # Get API key ID
        api_key_id = getattr(request.state, "api_key_id", None)
        if not api_key_id:
            return await call_next(request)

        # Determine required permission
        required_permission = self._get_required_permission(request.url.path, request.method)

        if required_permission:
            # Check permission
            if not self.signing_service.validate_api_key_permissions(api_key_id, required_permission):
                await report_security_event(
                    EventType.ACCESS_DENIED,
                    ThreatLevel.MEDIUM,
                    f"API key lacks required permission: {required_permission}",
                    details={
                        "api_key_id": api_key_id,
                        "required_permission": required_permission,
                        "path": request.url.path,
                        "method": request.method
                    }
                )

                raise HTTPException(
                    status_code=403,
                    detail=f"API key lacks required permission: {required_permission}"
                )

        return await call_next(request)

    def _get_required_permission(self, path: str, method: str) -> Optional[str]:
        """Determine required permission for path and method"""

        # Check path prefixes
        for path_prefix, permission in self.permission_map.items():
            if path.startswith(path_prefix):
                return permission

        # Default permissions based on method
        if method in ["GET", "HEAD", "OPTIONS"]:
            return "read"
        else:
            return "write"


# Export middleware classes
__all__ = [
    "RequestSigningMiddleware",
    "APIKeyPermissionMiddleware"
]
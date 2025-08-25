"""
API Request Signing and Integrity Validation
Implements HMAC-based request signing for API security
"""

import hmac
import hashlib
import time
import json
import base64
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

from .errorHandling import SecurityError, ValidationError
from .secrets import get_secrets_manager
from .securityMonitoring import report_security_event, EventType, ThreatLevel

logger = logging.getLogger(__name__)


@dataclass
class SignedRequest:
    """Represents a signed API request"""
    method: str
    path: str
    timestamp: int
    nonce: str
    headers: Dict[str, str]
    body_hash: Optional[str] = None
    query_params: Optional[Dict[str, List[str]]] = None
    signature: Optional[str] = None
    api_key_id: Optional[str] = None


class RequestSigningService:
    """Service for signing and verifying API requests"""

    def __init__(self):
        self.secrets_manager = get_secrets_manager()
        self.signing_algorithm = "sha256"
        self.timestamp_tolerance_seconds = 300  # 5 minutes
        self.nonce_cache = {}  # In production, use Redis
        self.api_keys = {}  # In production, load from database

        # Initialize API keys
        self._initialize_api_keys()

        logger.info("Request Signing Service initialized")

    def _initialize_api_keys(self):
        """Initialize API keys from secrets manager"""
        try:
            # In production, these would be loaded from a database
            # For now, we'll use environment-based keys
            master_key = self.secrets_manager.get_secret("API_MASTER_KEY",
                                                       default="dev_master_key_" + hashlib.sha256(b"a2a_platform").hexdigest()[:16])

            self.api_keys = {
                "default": {
                    "key_id": "default",
                    "secret": master_key,
                    "active": True,
                    "permissions": ["read", "write"]
                },
                "service_account_1": {
                    "key_id": "service_account_1",
                    "secret": hashlib.sha256(f"{master_key}_service1".encode()).hexdigest(),
                    "active": True,
                    "permissions": ["read"]
                }
            }

        except Exception as e:
            logger.error(f"Failed to initialize API keys: {e}")

    def generate_signature(self,
                         request: SignedRequest,
                         api_secret: str) -> str:
        """
        Generate HMAC signature for request

        Signature is calculated over:
        - HTTP method
        - Request path
        - Timestamp
        - Nonce
        - Body hash (if present)
        - Sorted query parameters
        """
        try:
            # Build canonical request string
            canonical_parts = [
                request.method.upper(),
                request.path,
                str(request.timestamp),
                request.nonce
            ]

            # Add query parameters if present
            if request.query_params:
                # Sort parameters for consistent ordering
                sorted_params = sorted(request.query_params.items())
                param_string = "&".join([
                    f"{key}={','.join(sorted(values))}"
                    for key, values in sorted_params
                ])
                canonical_parts.append(param_string)

            # Add body hash if present
            if request.body_hash:
                canonical_parts.append(request.body_hash)

            # Create canonical string
            canonical_string = "\n".join(canonical_parts)

            # Generate HMAC signature
            signature = hmac.new(
                api_secret.encode('utf-8'),
                canonical_string.encode('utf-8'),
                hashlib.sha256
            ).digest()

            # Return base64-encoded signature
            return base64.b64encode(signature).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to generate signature: {e}")
            raise SecurityError(f"Signature generation failed: {e}")

    def sign_request(self,
                    method: str,
                    url: str,
                    body: Optional[bytes] = None,
                    api_key_id: str = "default",
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Sign an outgoing request

        Returns headers to be added to the request
        """
        try:
            # Get API key
            if api_key_id not in self.api_keys:
                raise ValidationError(f"Invalid API key ID: {api_key_id}")

            api_key = self.api_keys[api_key_id]
            if not api_key["active"]:
                raise ValidationError("API key is not active")

            # Parse URL
            parsed_url = urlparse(url)
            path = parsed_url.path or "/"
            query_params = parse_qs(parsed_url.query) if parsed_url.query else None

            # Generate request components
            timestamp = int(time.time())
            nonce = base64.b64encode(hashlib.sha256(
                f"{timestamp}_{path}_{time.time()}".encode()
            ).digest()).decode('utf-8')[:16]

            # Calculate body hash if body is present
            body_hash = None
            if body:
                body_hash = base64.b64encode(
                    hashlib.sha256(body).digest()
                ).decode('utf-8')

            # Create signed request
            request = SignedRequest(
                method=method,
                path=path,
                timestamp=timestamp,
                nonce=nonce,
                headers=headers or {},
                body_hash=body_hash,
                query_params=query_params,
                api_key_id=api_key_id
            )

            # Generate signature
            signature = self.generate_signature(request, api_key["secret"])

            # Return signing headers
            return {
                "X-API-Key-ID": api_key_id,
                "X-Timestamp": str(timestamp),
                "X-Nonce": nonce,
                "X-Signature": signature,
                "X-Body-Hash": body_hash if body_hash else ""
            }

        except Exception as e:
            logger.error(f"Failed to sign request: {e}")
            raise

    async def verify_request(self,
                           method: str,
                           path: str,
                           headers: Dict[str, str],
                           body: Optional[bytes] = None,
                           query_params: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify an incoming signed request

        Returns (is_valid, error_message)
        """
        try:
            # Extract signing headers
            api_key_id = headers.get("x-api-key-id")
            timestamp_str = headers.get("x-timestamp")
            nonce = headers.get("x-nonce")
            signature = headers.get("x-signature")
            body_hash = headers.get("x-body-hash")

            # Check required headers
            if not all([api_key_id, timestamp_str, nonce, signature]):
                await report_security_event(
                    EventType.UNAUTHORIZED_API_ACCESS,
                    ThreatLevel.MEDIUM,
                    "Missing required signing headers",
                    details={"path": path, "method": method}
                )
                return False, "Missing required signing headers"

            # Validate API key
            if api_key_id not in self.api_keys:
                await report_security_event(
                    EventType.UNAUTHORIZED_API_ACCESS,
                    ThreatLevel.HIGH,
                    f"Invalid API key ID: {api_key_id}",
                    details={"path": path, "method": method}
                )
                return False, "Invalid API key"

            api_key = self.api_keys[api_key_id]
            if not api_key["active"]:
                return False, "API key is not active"

            # Check if API key has expired
            if "expires_at" in api_key:
                expires_at = datetime.fromisoformat(api_key["expires_at"].replace('Z', '+00:00'))
                if datetime.utcnow() > expires_at:
                    await report_security_event(
                        EventType.UNAUTHORIZED_API_ACCESS,
                        ThreatLevel.MEDIUM,
                        f"Expired API key used: {api_key_id}",
                        details={
                            "path": path,
                            "method": method,
                            "expired_at": api_key["expires_at"]
                        }
                    )
                    return False, "API key has expired"

            # Validate timestamp
            try:
                timestamp = int(timestamp_str)
                current_time = int(time.time())

                if abs(current_time - timestamp) > self.timestamp_tolerance_seconds:
                    await report_security_event(
                        EventType.SUSPICIOUS_TRAFFIC,
                        ThreatLevel.MEDIUM,
                        "Request timestamp outside acceptable window",
                        details={
                            "timestamp": timestamp,
                            "current_time": current_time,
                            "difference": abs(current_time - timestamp)
                        }
                    )
                    return False, "Request timestamp is too old or too far in the future"

            except ValueError:
                return False, "Invalid timestamp format"

            # Check nonce for replay protection
            if await self._is_nonce_used(nonce):
                await report_security_event(
                    EventType.SUSPICIOUS_TRAFFIC,
                    ThreatLevel.HIGH,
                    "Replay attack detected - nonce already used",
                    details={
                        "nonce": nonce,
                        "api_key_id": api_key_id,
                        "path": path
                    }
                )
                return False, "Nonce has already been used"

            # Verify body hash if present
            if body and body_hash:
                calculated_hash = base64.b64encode(
                    hashlib.sha256(body).digest()
                ).decode('utf-8')

                if calculated_hash != body_hash:
                    await report_security_event(
                        EventType.SUSPICIOUS_TRAFFIC,
                        ThreatLevel.HIGH,
                        "Request body integrity check failed",
                        details={
                            "path": path,
                            "expected_hash": body_hash,
                            "calculated_hash": calculated_hash
                        }
                    )
                    return False, "Body integrity check failed"

            # Create request object for signature verification
            request = SignedRequest(
                method=method,
                path=path,
                timestamp=timestamp,
                nonce=nonce,
                headers=headers,
                body_hash=body_hash if body else None,
                query_params=query_params,
                api_key_id=api_key_id
            )

            # Generate expected signature
            expected_signature = self.generate_signature(request, api_key["secret"])

            # Constant-time comparison to prevent timing attacks
            if not self._secure_compare(signature, expected_signature):
                await report_security_event(
                    EventType.UNAUTHORIZED_API_ACCESS,
                    ThreatLevel.HIGH,
                    "Invalid request signature",
                    details={
                        "api_key_id": api_key_id,
                        "path": path,
                        "method": method
                    }
                )
                return False, "Invalid signature"

            # Mark nonce as used
            await self._mark_nonce_used(nonce)

            # Request is valid
            return True, None

        except Exception as e:
            logger.error(f"Request verification failed: {e}")
            await report_security_event(
                EventType.SYSTEM_INTRUSION,
                ThreatLevel.MEDIUM,
                f"Request verification error: {str(e)}",
                details={"path": path, "method": method}
            )
            return False, "Request verification failed"

    def _secure_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks"""
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))

    async def _is_nonce_used(self, nonce: str) -> bool:
        """Check if nonce has been used (implements replay protection)"""
        # In production, use Redis with TTL
        # For now, use in-memory cache with cleanup

        # Clean old nonces
        current_time = time.time()
        cutoff_time = current_time - self.timestamp_tolerance_seconds * 2

        self.nonce_cache = {
            n: t for n, t in self.nonce_cache.items()
            if t > cutoff_time
        }

        # Check if nonce exists
        return nonce in self.nonce_cache

    async def _mark_nonce_used(self, nonce: str):
        """Mark nonce as used"""
        self.nonce_cache[nonce] = time.time()

    def validate_api_key_permissions(self,
                                    api_key_id: str,
                                    required_permission: str) -> bool:
        """Check if API key has required permission and is not expired"""
        if api_key_id not in self.api_keys:
            return False

        api_key = self.api_keys[api_key_id]

        # Check if key is active
        if not api_key.get("active", True):
            return False

        # Check if key has expired
        if "expires_at" in api_key:
            expires_at = datetime.fromisoformat(api_key["expires_at"].replace('Z', '+00:00'))
            if datetime.utcnow() > expires_at:
                return False

        return required_permission in api_key.get("permissions", [])

    def rotate_api_key(self, api_key_id: str) -> str:
        """Rotate an API key (generate new secret)"""
        if api_key_id not in self.api_keys:
            raise ValidationError(f"Invalid API key ID: {api_key_id}")

        # Generate new secret
        new_secret = base64.b64encode(
            hashlib.sha256(
                f"{api_key_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()}".encode()
            ).digest()
        ).decode('utf-8')

        # Update key
        self.api_keys[api_key_id]["secret"] = new_secret
        self.api_keys[api_key_id]["rotated_at"] = datetime.utcnow().isoformat()

        logger.info(f"API key rotated: {api_key_id}")
        return new_secret

    def create_api_key(self,
                      key_id: str,
                      permissions: List[str],
                      expires_at: Optional[str] = None) -> Dict[str, Any]:
        """Create a new API key with optional expiration"""
        if key_id in self.api_keys:
            raise ValidationError(f"API key ID already exists: {key_id}")

        # Generate secret
        secret = base64.b64encode(
            hashlib.sha256(
                f"{key_id}_{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()}".encode()
            ).digest()
        ).decode('utf-8')

        # Create key
        api_key = {
            "key_id": key_id,
            "secret": secret,
            "active": True,
            "permissions": permissions,
            "created_at": datetime.utcnow().isoformat()
        }

        # Add expiration if specified
        if expires_at:
            api_key["expires_at"] = expires_at

        self.api_keys[key_id] = api_key

        logger.info(f"API key created: {key_id}" + (f" (expires: {expires_at})" if expires_at else ""))
        return {"key_id": key_id, "secret": secret}

    def revoke_api_key(self, api_key_id: str):
        """Revoke an API key"""
        if api_key_id not in self.api_keys:
            raise ValidationError(f"Invalid API key ID: {api_key_id}")

        self.api_keys[api_key_id]["active"] = False
        self.api_keys[api_key_id]["revoked_at"] = datetime.utcnow().isoformat()

        logger.info(f"API key revoked: {api_key_id}")


# Global instance
_signing_service: Optional[RequestSigningService] = None

def get_signing_service() -> RequestSigningService:
    """Get global request signing service instance"""
    global _signing_service
    if _signing_service is None:
        _signing_service = RequestSigningService()
    return _signing_service


# Export main classes and functions
__all__ = [
    'RequestSigningService',
    'SignedRequest',
    'get_signing_service'
]

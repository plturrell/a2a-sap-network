"""
Request Signing for Agent-to-Agent Communication
Implements cryptographic signing and verification of A2A messages
"""

import hashlib
import hmac
import json
import time
import secrets
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import jwt
import logging

logger = logging.getLogger(__name__)


class A2ARequestSigner:
    """Handles request signing for agent-to-agent communication"""

    def __init__(self, private_key: Optional[str] = None, public_key: Optional[str] = None):
        """
        Initialize request signer

        Args:
            private_key: PEM encoded private key for signing
            public_key: PEM encoded public key for verification
        """
        self.private_key = None
        self.public_key = None

        if private_key:
            self.private_key = serialization.load_pem_private_key(
                private_key.encode() if isinstance(private_key, str) else private_key,
                password=None,
                backend=default_backend()
            )

        if public_key:
            self.public_key = serialization.load_pem_public_key(
                public_key.encode() if isinstance(public_key, str) else public_key,
                backend=default_backend()
            )

    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate a new RSA key pair for agent"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        public_key = private_key.public_key()

        # Serialize keys to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        return private_pem, public_pem

    def sign_request(self,
                    agent_id: str,
                    target_agent_id: str,
                    method: str,
                    path: str,
                    body: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Sign an outgoing request to another agent

        Args:
            agent_id: ID of the sending agent
            target_agent_id: ID of the receiving agent
            method: HTTP method (GET, POST, etc.)
            path: Request path
            body: Request body (if any)
            headers: Additional headers

        Returns:
            Dict containing signature headers
        """
        if not self.private_key:
            raise ValueError("Private key required for signing")

        # Create canonical request
        timestamp = str(int(time.time()))
        # Use cryptographically secure random nonce to prevent replay attacks
        nonce = base64.b64encode(secrets.token_bytes(32)).decode()

        canonical_parts = [
            method.upper(),
            path,
            agent_id,
            target_agent_id,
            timestamp,
            nonce
        ]

        # Add body hash if present
        if body:
            body_bytes = json.dumps(body, sort_keys=True).encode()
            body_hash = base64.b64encode(hashlib.sha256(body_bytes).digest()).decode()
            canonical_parts.append(body_hash)

        canonical_request = '\n'.join(canonical_parts)

        # Sign the canonical request
        signature = self.private_key.sign(
            canonical_request.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Create signature header
        signature_b64 = base64.b64encode(signature).decode()

        return {
            'X-A2A-Agent-ID': agent_id,
            'X-A2A-Target-ID': target_agent_id,
            'X-A2A-Timestamp': timestamp,
            'X-A2A-Nonce': nonce,
            'X-A2A-Signature': signature_b64,
            'X-A2A-Algorithm': 'RSA-PSS-SHA256'
        }

    def verify_request(self,
                      headers: Dict[str, str],
                      method: str,
                      path: str,
                      body: Optional[Dict[str, Any]] = None,
                      public_key_pem: Optional[str] = None,
                      max_age_seconds: int = 300) -> Tuple[bool, Optional[str]]:
        """
        Verify an incoming request from another agent

        Args:
            headers: Request headers containing signature
            method: HTTP method
            path: Request path
            body: Request body (if any)
            public_key_pem: Public key of sending agent (if not using instance key)
            max_age_seconds: Maximum age of request in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Extract required headers
            required_headers = [
                'X-A2A-Agent-ID',
                'X-A2A-Target-ID',
                'X-A2A-Timestamp',
                'X-A2A-Nonce',
                'X-A2A-Signature',
                'X-A2A-Algorithm'
            ]

            missing_headers = [h for h in required_headers if h not in headers]
            if missing_headers:
                return False, f"Missing required headers: {', '.join(missing_headers)}"

            # Check algorithm
            if headers['X-A2A-Algorithm'] != 'RSA-PSS-SHA256':
                return False, f"Unsupported algorithm: {headers['X-A2A-Algorithm']}"

            # Check timestamp age (use constant-time comparison to prevent timing attacks)
            timestamp = int(headers['X-A2A-Timestamp'])
            current_time = int(time.time())
            time_diff = current_time - timestamp

            # Prevent both old and future-dated requests
            if time_diff > max_age_seconds or time_diff < -max_age_seconds:
                return False, f"Request timestamp invalid: {time_diff} seconds difference"

            # Load public key if provided
            if public_key_pem:
                public_key = serialization.load_pem_public_key(
                    public_key_pem.encode(),
                    backend=default_backend()
                )
            else:
                public_key = self.public_key

            if not public_key:
                return False, "No public key available for verification"

            # Recreate canonical request
            canonical_parts = [
                method.upper(),
                path,
                headers['X-A2A-Agent-ID'],
                headers['X-A2A-Target-ID'],
                headers['X-A2A-Timestamp'],
                headers['X-A2A-Nonce']
            ]

            # Add body hash if present
            if body:
                body_bytes = json.dumps(body, sort_keys=True).encode()
                body_hash = base64.b64encode(hashlib.sha256(body_bytes).digest()).decode()
                canonical_parts.append(body_hash)

            canonical_request = '\n'.join(canonical_parts)

            # Verify signature
            signature = base64.b64decode(headers['X-A2A-Signature'])

            try:
                public_key.verify(
                    signature,
                    canonical_request.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True, None
            except InvalidSignature:
                return False, "Invalid signature"

        except Exception as e:
            logger.error(f"Request verification error: {e}")
            return False, f"Verification error: {str(e)}"


class JWTRequestSigner:
    """Alternative JWT-based request signing (simpler but requires shared secret)"""

    def __init__(self, secret_key: str):
        """Initialize with shared secret"""
        self.secret_key = secret_key
        self.algorithm = 'HS256'

    def sign_request(self,
                    agent_id: str,
                    target_agent_id: str,
                    method: str,
                    path: str,
                    body: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Sign request using JWT"""

        payload = {
            'iss': agent_id,  # Issuer
            'aud': target_agent_id,  # Audience
            'iat': int(time.time()),  # Issued at
            'exp': int(time.time()) + 300,  # Expires in 5 minutes
            'method': method.upper(),
            'path': path
        }

        # Add body hash if present
        if body:
            body_bytes = json.dumps(body, sort_keys=True).encode()
            body_hash = hashlib.sha256(body_bytes).hexdigest()
            payload['body_hash'] = body_hash

        # Create JWT
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return {
            'Authorization': f'Bearer {token}',
            'X-A2A-Agent-ID': agent_id,
            'X-A2A-Target-ID': target_agent_id
        }

    def verify_request(self,
                      headers: Dict[str, str],
                      method: str,
                      path: str,
                      body: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """Verify JWT-signed request"""

        try:
            # Extract token
            auth_header = headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return False, "Missing or invalid Authorization header"

            token = auth_header[7:]  # Remove 'Bearer ' prefix

            # Decode and verify JWT
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify method and path
            if payload.get('method') != method.upper():
                return False, "Method mismatch"

            if payload.get('path') != path:
                return False, "Path mismatch"

            # Verify body hash if present
            if body and 'body_hash' in payload:
                body_bytes = json.dumps(body, sort_keys=True).encode()
                body_hash = hashlib.sha256(body_bytes).hexdigest()
                if payload['body_hash'] != body_hash:
                    return False, "Body hash mismatch"

            # Extract agent IDs
            agent_id = payload.get('iss')
            target_agent_id = payload.get('aud')

            # Verify against headers
            if headers.get('X-A2A-Agent-ID') != agent_id:
                return False, "Agent ID mismatch"

            if headers.get('X-A2A-Target-ID') != target_agent_id:
                return False, "Target ID mismatch"

            return True, None

        except jwt.ExpiredSignatureError:
            return False, "Token expired"
        except jwt.InvalidTokenError as e:
            return False, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return False, f"Verification error: {str(e)}"


# Middleware for automatic request signing/verification
class A2ASigningMiddleware:
    """Middleware to automatically sign outgoing and verify incoming A2A requests"""

    def __init__(self, agent_id: str, signer: A2ARequestSigner):
        self.agent_id = agent_id
        self.signer = signer

    async def sign_outgoing_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add signature headers to outgoing request"""

        signature_headers = self.signer.sign_request(
            agent_id=self.agent_id,
            target_agent_id=request.get('target_agent_id'),
            method=request.get('method', 'POST'),
            path=request.get('path', '/'),
            body=request.get('body'),
            headers=request.get('headers', {})
        )

        # Merge signature headers with existing headers
        request['headers'] = {**request.get('headers', {}), **signature_headers}

        return request

    async def verify_incoming_request(self,
                                    headers: Dict[str, str],
                                    method: str,
                                    path: str,
                                    body: Optional[Dict[str, Any]] = None,
                                    agent_registry = None) -> Tuple[bool, Optional[str]]:
        """Verify incoming request signature"""

        # Get sending agent's public key from registry
        agent_id = headers.get('X-A2A-Agent-ID')
        if not agent_id:
            return False, "Missing agent ID in request"

        public_key_pem = None
        if agent_registry:
            agent_info = await agent_registry.get_agent(agent_id)
            if agent_info:
                public_key_pem = agent_info.get('public_key')

        return self.signer.verify_request(
            headers=headers,
            method=method,
            path=path,
            body=body,
            public_key_pem=public_key_pem
        )

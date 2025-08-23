"""
Production-Ready Authentication Middleware for MCP Servers
Implements JWT-based authentication with API key support
"""

import os
import jwt
import time
import hashlib
import secrets
import logging
from typing import Dict, Optional, Any, Callable
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MCPAuthMiddleware:
    """Commercial-grade authentication middleware for MCP servers"""
    
    def __init__(self):
        # Load from environment with secure defaults
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', self._generate_secure_secret())
        self.jwt_algorithm = 'HS256'
        self.jwt_expiration = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
        self.api_key_header = os.getenv('API_KEY_HEADER', 'X-API-Key')
        
        # API key storage - in production, use database
        self.api_keys = self._load_api_keys()
        
        # Rate limiting configuration
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
        self.rate_limit_cache: Dict[str, list] = {}
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
    def _generate_secure_secret(self) -> str:
        """Generate cryptographically secure secret"""
        return secrets.token_urlsafe(32)
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from secure storage"""
        # In production, load from database or secret manager
        # For now, using environment variables
        api_keys = {}
        
        # Example API key format: API_KEY_1=key:client_id:permissions
        for key, value in os.environ.items():
            if key.startswith('API_KEY_'):
                try:
                    api_key, client_id, permissions = value.split(':')
                    api_keys[api_key] = {
                        'client_id': client_id,
                        'permissions': permissions.split(','),
                        'created_at': datetime.utcnow(),
                        'last_used': None
                    }
                except ValueError:
                    logger.warning(f"Invalid API key format for {key}")
                    
        return api_keys
    
    def generate_jwt_token(self, user_id: str, permissions: list) -> str:
        """Generate JWT token with user claims"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiration),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return client info"""
        if api_key in self.api_keys:
            client_info = self.api_keys[api_key]
            # Update last used timestamp
            client_info['last_used'] = datetime.utcnow()
            return client_info
        return None
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        # Clean old entries
        if client_id in self.rate_limit_cache:
            self.rate_limit_cache[client_id] = [
                timestamp for timestamp in self.rate_limit_cache[client_id]
                if current_time - timestamp < self.rate_limit_window
            ]
        
        # Check current rate
        if client_id not in self.rate_limit_cache:
            self.rate_limit_cache[client_id] = []
        
        if len(self.rate_limit_cache[client_id]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.rate_limit_cache[client_id].append(current_time)
        return True
    
    def authenticate_request(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Authenticate request using JWT or API key"""
        # Check for JWT token
        auth_header = headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            jwt_payload = self.verify_jwt_token(token)
            if jwt_payload:
                return {
                    'type': 'jwt',
                    'client_id': jwt_payload['user_id'],
                    'permissions': jwt_payload['permissions'],
                    'payload': jwt_payload
                }
        
        # Check for API key
        api_key = headers.get(self.api_key_header, '')
        if api_key:
            client_info = self.verify_api_key(api_key)
            if client_info:
                return {
                    'type': 'api_key',
                    'client_id': client_info['client_id'],
                    'permissions': client_info['permissions'],
                    'info': client_info
                }
        
        return None
    
    def require_auth(self, permissions: Optional[list] = None):
        """Decorator for protecting endpoints"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request object (FastAPI specific)
                request = kwargs.get('request')
                if not request:
                    # Try to find request in args
                    for arg in args:
                        if hasattr(arg, 'headers'):
                            request = arg
                            break
                
                if not request:
                    return {
                        'error': 'Authentication required',
                        'status': 'unauthorized',
                        'code': 401
                    }
                
                # Convert headers to dict
                headers = dict(request.headers)
                
                # Authenticate
                auth_info = self.authenticate_request(headers)
                if not auth_info:
                    return {
                        'error': 'Invalid or missing credentials',
                        'status': 'unauthorized',
                        'code': 401
                    }
                
                # Check rate limit
                if not self.check_rate_limit(auth_info['client_id']):
                    return {
                        'error': 'Rate limit exceeded',
                        'status': 'rate_limited',
                        'code': 429
                    }
                
                # Check permissions
                if permissions:
                    user_permissions = auth_info.get('permissions', [])
                    if not any(perm in user_permissions for perm in permissions):
                        return {
                            'error': 'Insufficient permissions',
                            'status': 'forbidden',
                            'code': 403
                        }
                
                # Add auth info to kwargs
                kwargs['auth_info'] = auth_info
                
                # Call original function
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def add_security_headers(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add security headers to response"""
        if '_headers' not in response:
            response['_headers'] = {}
        
        response['_headers'].update(self.security_headers)
        return response

# Global middleware instance
mcp_auth = MCPAuthMiddleware()

# Export decorators and functions
require_auth = mcp_auth.require_auth
generate_token = mcp_auth.generate_jwt_token
verify_token = mcp_auth.verify_jwt_token
"""
Secure A2A Agent Base Class
Provides enhanced security features for all A2A agents
"""

import logging
import time
import re
import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
from functools import wraps

from app.a2a.sdk.agentBase import A2AAgentBase
from app.a2a.sdk.types import A2AMessage, MessageRole
from app.a2a.sdk.utils import create_error_response, create_success_response

logger = logging.getLogger(__name__)


class SecureA2AAgent(A2AAgentBase):
    """
    Enhanced secure base class for A2A agents with built-in security features:
    - Input validation and sanitization
    - Rate limiting
    - Authentication and authorization
    - Encrypted communication
    - Audit logging
    - Security monitoring
    """

    def __init__(self, config):
        super().__init__(config)

        # Security configuration
        self.security_config = {
            'enable_rate_limiting': True,
            'enable_input_validation': True,
            'enable_audit_logging': True,
            'enable_encryption': True,
            'max_request_size': 1024 * 1024,  # 1MB
            'session_timeout': 3600,  # 1 hour
            'max_failed_auth_attempts': 5,
            'auth_lockout_duration': 300  # 5 minutes
        }

        # Rate limiting
        self.rate_limiters = {}
        self.default_rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'burst_size': 10
        }

        # Authentication tracking
        self.auth_attempts = defaultdict(lambda: {'count': 0, 'last_attempt': None})
        self.active_sessions = {}

        # Audit logger
        self.audit_logger = logging.getLogger(f'{self.__class__.__name__}.audit')
        self.security_logger = logging.getLogger(f'{self.__class__.__name__}.security')

        # Input validation patterns
        self.validation_patterns = {
            'sql_injection': re.compile(
                r"((SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM)|--|;|'|\"|\\*|OR\\s+1=1|AND\\s+1=1)",
                re.IGNORECASE
            ),
            'xss_attack': re.compile(
                r"(<script|<iframe|javascript:|onerror=|onload=|onclick=|<img\s+src)",
                re.IGNORECASE
            ),
            'path_traversal': re.compile(r"(\.\./|\.\.\\|%2e%2e|%252e%252e)"),
            'command_injection': re.compile(r"[;&|`$(){}\[\]\n\r]"),
            'ldap_injection': re.compile(r"[\(\)\*\\]")
        }

        # Encryption keys (in production, use proper key management)
        self._init_encryption()

        logger.info(f"Initialized SecureA2AAgent with enhanced security features")

    def _init_encryption(self):
        """Initialize encryption for secure communication"""
        try:
            from cryptography.fernet import Fernet
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
            self.encryption_enabled = True
        except ImportError:
            logger.warning("Cryptography library not available - encryption disabled")
            self.encryption_enabled = False

    def secure_handler(self, handler_func):
        """Decorator to add security checks to message handlers"""
        @wraps(handler_func)
        async def wrapper(message: A2AMessage, context_id: str, **kwargs):
            try:
                # Extract request data
                request_data = {}
                for part in message.parts:
                    if part.data:
                        request_data.update(part.data)

                # 1. Rate limiting check
                client_id = request_data.get('client_id', message.metadata.get('sender_id', 'unknown'))
                if not self._check_rate_limit(client_id):
                    self.security_logger.warning(f"Rate limit exceeded for client: {client_id}")
                    return create_error_response("Rate limit exceeded", code="RATE_LIMIT_EXCEEDED")

                # 2. Input validation
                validation_result = self._validate_input(request_data)
                if not validation_result['valid']:
                    self.security_logger.warning(f"Input validation failed: {validation_result['reason']}")
                    return create_error_response(f"Invalid input: {validation_result['reason']}", code="INVALID_INPUT")

                # 3. Authentication check
                auth_result = await self._check_authentication(message, request_data)
                if not auth_result['authenticated']:
                    self.security_logger.warning(f"Authentication failed: {auth_result['reason']}")
                    return create_error_response("Authentication required", code="AUTH_REQUIRED")

                # 4. Authorization check
                if not await self._check_authorization(message, handler_func.__name__):
                    self.security_logger.warning(f"Authorization failed for handler: {handler_func.__name__}")
                    return create_error_response("Insufficient permissions", code="FORBIDDEN")

                # 5. Audit logging
                self._audit_log('request', {
                    'handler': handler_func.__name__,
                    'client_id': client_id,
                    'message_id': message.id,
                    'timestamp': datetime.utcnow().isoformat()
                })

                # Execute the actual handler
                result = await handler_func(message, context_id, **kwargs)

                # 6. Audit response
                self._audit_log('response', {
                    'handler': handler_func.__name__,
                    'client_id': client_id,
                    'message_id': message.id,
                    'success': result.get('status') == 'success',
                    'timestamp': datetime.utcnow().isoformat()
                })

                return result

            except Exception as e:
                self.security_logger.error(f"Security handler error: {e}")
                return create_error_response("Internal security error", code="SECURITY_ERROR")

        return wrapper

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limits"""
        if not self.security_config['enable_rate_limiting']:
            return True

        current_time = time.time()

        if client_id not in self.rate_limiters:
            self.rate_limiters[client_id] = {
                'minute_window': current_time,
                'minute_count': 0,
                'hour_window': current_time,
                'hour_count': 0
            }

        limiter = self.rate_limiters[client_id]

        # Check minute window
        if current_time - limiter['minute_window'] > 60:
            limiter['minute_window'] = current_time
            limiter['minute_count'] = 0

        # Check hour window
        if current_time - limiter['hour_window'] > 3600:
            limiter['hour_window'] = current_time
            limiter['hour_count'] = 0

        # Check limits
        if limiter['minute_count'] >= self.default_rate_limits['requests_per_minute']:
            return False

        if limiter['hour_count'] >= self.default_rate_limits['requests_per_hour']:
            return False

        # Update counts
        limiter['minute_count'] += 1
        limiter['hour_count'] += 1

        return True

    def _validate_input(self, data: Any) -> Dict[str, Any]:
        """Validate input data for security threats"""
        if not self.security_config['enable_input_validation']:
            return {'valid': True}

        def check_value(value: Any, path: str = '') -> Optional[str]:
            if isinstance(value, str):
                # Check for various injection attacks
                for attack_type, pattern in self.validation_patterns.items():
                    if pattern.search(value):
                        return f"{attack_type} pattern detected at {path}"

                # Check string length
                if len(value) > 10000:
                    return f"String too long at {path}"

            elif isinstance(value, dict):
                # Check dictionary depth
                if path.count('.') > 10:
                    return f"Object too deeply nested at {path}"

                # Recursively check all values
                for k, v in value.items():
                    error = check_value(v, f"{path}.{k}" if path else k)
                    if error:
                        return error

            elif isinstance(value, list):
                # Check array size
                if len(value) > 1000:
                    return f"Array too large at {path}"

                # Check all items
                for i, item in enumerate(value):
                    error = check_value(item, f"{path}[{i}]")
                    if error:
                        return error

            return None

        error = check_value(data)
        if error:
            return {'valid': False, 'reason': error}

        return {'valid': True}

    async def _check_authentication(self, message: A2AMessage, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the request is properly authenticated"""
        # Extract auth token
        auth_token = request_data.get('auth_token') or message.metadata.get('auth_token')

        if not auth_token:
            return {'authenticated': False, 'reason': 'No auth token provided'}

        # Validate token (simplified - in production use proper JWT validation)
        if auth_token in self.active_sessions:
            session = self.active_sessions[auth_token]
            if time.time() - session['created'] < self.security_config['session_timeout']:
                return {'authenticated': True, 'session': session}
            else:
                # Session expired
                del self.active_sessions[auth_token]
                return {'authenticated': False, 'reason': 'Session expired'}

        # For A2A protocol, we trust the blockchain-verified sender
        if message.metadata.get('blockchain_verified'):
            return {'authenticated': True, 'blockchain_verified': True}

        return {'authenticated': False, 'reason': 'Invalid auth token'}

    async def _check_authorization(self, message: A2AMessage, handler_name: str) -> bool:
        """Check if the authenticated user has permission for this handler"""
        # For A2A agents, check if the sender agent has permission
        sender_id = message.metadata.get('sender_id')

        # Define handler permissions (simplified)
        public_handlers = ['health_check', 'get_info', 'ping']
        admin_handlers = ['update_config', 'shutdown', 'reset']

        if handler_name in public_handlers:
            return True

        if handler_name in admin_handlers:
            # Check if sender is an admin agent
            return sender_id in self.config.get('admin_agents', [])

        # Default: allow authenticated agents
        return True

    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Log security-relevant events for audit trail"""
        if not self.security_config['enable_audit_logging']:
            return

        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'agent_id': self.agent_id,
            'details': details
        }

        self.audit_logger.info(json.dumps(audit_entry))

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        if not self.encryption_enabled:
            return data

        return self.cipher_suite.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        if not self.encryption_enabled:
            return encrypted_data

        return self.cipher_suite.decrypt(encrypted_data)

    def generate_session_token(self, client_id: str) -> str:
        """Generate a secure session token"""
        token_data = f"{client_id}:{time.time()}:{os.urandom(16).hex()}"
        token = hashlib.sha256(token_data.encode()).hexdigest()

        self.active_sessions[token] = {
            'client_id': client_id,
            'created': time.time(),
            'last_activity': time.time()
        }

        return token

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and return session information"""
        if token in self.active_sessions:
            session = self.active_sessions[token]
            if time.time() - session['created'] < self.security_config['session_timeout']:
                session['last_activity'] = time.time()
                return session
            else:
                del self.active_sessions[token]

        return None

    async def security_scan(self) -> Dict[str, Any]:
        """Perform security self-scan"""
        scan_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'security_features': {
                'rate_limiting': self.security_config['enable_rate_limiting'],
                'input_validation': self.security_config['enable_input_validation'],
                'audit_logging': self.security_config['enable_audit_logging'],
                'encryption': self.encryption_enabled
            },
            'active_sessions': len(self.active_sessions),
            'rate_limit_tracking': len(self.rate_limiters),
            'recent_auth_failures': sum(
                1 for attempts in self.auth_attempts.values()
                if attempts['count'] > 0
            ),
            'vulnerabilities': []
        }

        # Check for potential vulnerabilities
        if not self.encryption_enabled:
            scan_results['vulnerabilities'].append({
                'type': 'missing_encryption',
                'severity': 'medium',
                'description': 'Encryption is not enabled'
            })

        if not self.security_config['enable_rate_limiting']:
            scan_results['vulnerabilities'].append({
                'type': 'no_rate_limiting',
                'severity': 'high',
                'description': 'Rate limiting is disabled'
            })

        return scan_results

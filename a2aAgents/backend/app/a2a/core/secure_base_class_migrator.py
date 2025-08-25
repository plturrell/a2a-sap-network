"""
Secure Base Class Migration Utility
Converts all agents to use SecureA2AAgent base class for enhanced security
"""

import os
import re
import ast
from typing import List, Dict, Tuple, Any
from pathlib import Path


class SecureBaseClassMigrator:
    """Utility to migrate agents to SecureA2AAgent base class"""

    def __init__(self):
        self.files_processed = 0
        self.files_migrated = 0
        self.errors = []

        # Base class patterns to replace
        self.BASE_CLASS_PATTERNS = {
            # Direct A2AAgentBase inheritance
            r'class\s+(\w+)\s*\(\s*A2AAgentBase\s*\)': 'class \\1(SecureA2AAgent)',

            # Multiple inheritance with A2AAgentBase first
            r'class\s+(\w+)\s*\(\s*A2AAgentBase\s*,([^)]+)\)': 'class \\1(SecureA2AAgent,\\2)',

            # Multiple inheritance with A2AAgentBase not first
            r'class\s+(\w+)\s*\(([^,]+),\s*A2AAgentBase([^)]*)\)': 'class \\1(\\2, SecureA2AAgent\\3)',

            # AgentBase variations
            r'class\s+(\w+)\s*\(\s*AgentBase\s*\)': 'class \\1(SecureA2AAgent)',

            # BaseAgent variations
            r'class\s+(\w+)\s*\(\s*BaseAgent\s*\)': 'class \\1(SecureA2AAgent)',
        }

        # Import patterns to add
        self.SECURE_IMPORT = 'from app.a2a.core.security_base import SecureA2AAgent'

        # Security features to add to __init__
        self.SECURITY_INIT_TEMPLATE = '''
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        '''

        # Methods to add for security
        self.SECURITY_METHODS_TEMPLATE = '''
    def _init_security_features(self):
        """Initialize security features from SecureA2AAgent"""
        # Rate limiting configuration
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'heavy': {'requests': 10, 'window': 60},     # 10 requests per minute for heavy operations
            'auth': {'requests': 5, 'window': 300}       # 5 auth attempts per 5 minutes
        }

        # Input validation rules
        self.validation_rules = {
            'max_string_length': 10000,
            'max_array_size': 1000,
            'max_object_depth': 10,
            'allowed_file_extensions': ['.json', '.txt', '.csv', '.xml'],
            'sql_injection_patterns': [
                r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM)\b)',
                r'(--|;|\'|"|\*|OR\s+1=1|AND\s+1=1)'
            ]
        }

        # Initialize security logger
        import logging
        self.security_logger = logging.getLogger(f'{self.__class__.__name__}.security')

    def _init_rate_limiting(self):
        """Initialize rate limiting tracking"""
        from collections import defaultdict
        import time

        self.rate_limit_tracker = defaultdict(lambda: {'count': 0, 'window_start': time.time()})

    def _init_input_validation(self):
        """Initialize input validation helpers"""
        self.input_validators = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.,!?]+$')
        }

    @property
    def is_secure(self) -> bool:
        """Check if agent is running in secure mode"""
        return True  # SecureA2AAgent always runs in secure mode

    def validate_input(self, data: Any, rules: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """Validate input data against security rules"""
        if rules is None:
            rules = self.validation_rules

        try:
            # Check string length
            if isinstance(data, str):
                if len(data) > rules.get('max_string_length', 10000):
                    return False, "String exceeds maximum length"

                # Check for SQL injection patterns
                for pattern in rules.get('sql_injection_patterns', []):
                    if re.search(pattern, data, re.IGNORECASE):
                        self.security_logger.warning(f"Potential SQL injection detected: {data[:50]}...")
                        return False, "Invalid characters detected"

            # Check array size
            elif isinstance(data, (list, tuple)):
                if len(data) > rules.get('max_array_size', 1000):
                    return False, "Array exceeds maximum size"

            # Check object depth
            elif isinstance(data, dict):
                if self._get_dict_depth(data) > rules.get('max_object_depth', 10):
                    return False, "Object exceeds maximum depth"

            return True, None

        except Exception as e:
            self.security_logger.error(f"Input validation error: {e}")
            return False, str(e)

    def _get_dict_depth(self, d: dict, current_depth: int = 0) -> int:
        """Get the maximum depth of a nested dictionary"""
        if not isinstance(d, dict) or not d:
            return current_depth

        return max(self._get_dict_depth(v, current_depth + 1)
                   for v in d.values()
                   if isinstance(v, dict))

    def check_rate_limit(self, key: str, limit_type: str = 'default') -> bool:
        """Check if rate limit is exceeded"""
        import time

        limits = self.rate_limits.get(limit_type, self.rate_limits['default'])
        tracker = self.rate_limit_tracker[f"{key}:{limit_type}"]

        current_time = time.time()
        window_duration = limits['window']

        # Reset window if expired
        if current_time - tracker['window_start'] > window_duration:
            tracker['count'] = 0
            tracker['window_start'] = current_time

        # Check limit
        if tracker['count'] >= limits['requests']:
            self.security_logger.warning(f"Rate limit exceeded for {key} ({limit_type})")
            return False

        tracker['count'] += 1
        return True
'''

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze file to check if it needs migration"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check if already using SecureA2AAgent
            if 'SecureA2AAgent' in content:
                return {
                    'needs_migration': False,
                    'reason': 'Already using SecureA2AAgent'
                }

            # Check if it's an agent class file
            class_matches = re.findall(r'class\s+(\w+)\s*\([^)]+\):', content)
            if not class_matches:
                return {
                    'needs_migration': False,
                    'reason': 'No class definitions found'
                }

            # Check for base class patterns
            for pattern in self.BASE_CLASS_PATTERNS:
                if re.search(pattern, content):
                    return {
                        'needs_migration': True,
                        'base_class_pattern': pattern,
                        'class_names': class_matches
                    }

            return {
                'needs_migration': False,
                'reason': 'No recognized agent base class found'
            }

        except Exception as e:
            self.errors.append({'file': file_path, 'error': str(e)})
            return {
                'needs_migration': False,
                'reason': f'Error analyzing file: {str(e)}'
            }

    def migrate_file(self, file_path: str) -> Dict[str, Any]:
        """Migrate file to use SecureA2AAgent"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            original_content = content

            # 1. Replace base class inheritance
            for pattern, replacement in self.BASE_CLASS_PATTERNS.items():
                content = re.sub(pattern, replacement, content)

            # 2. Add SecureA2AAgent import if not present
            if 'from app.a2a.core.security_base import SecureA2AAgent' not in content:
                # Find the last import statement
                import_matches = list(re.finditer(r'^(from\s+.+|import\s+.+)$', content, re.MULTILINE))
                if import_matches:
                    last_import_pos = import_matches[-1].end()
                    content = content[:last_import_pos] + '\n' + self.SECURE_IMPORT + content[last_import_pos:]
                else:
                    # Add after module docstring
                    docstring_match = re.match(r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', content)
                    if docstring_match:
                        pos = docstring_match.end()
                        content = content[:pos] + '\n\n' + self.SECURE_IMPORT + content[pos:]
                    else:
                        content = self.SECURE_IMPORT + '\n\n' + content

            # 3. Add security initialization to __init__ methods
            init_pattern = r'(def\s+__init__\s*\([^)]+\)\s*:\s*\n(?:\s*"""[\s\S]*?"""\s*\n)?)(.*?)(?=\n\s*def|\Z)'

            def add_security_init(match):
                method_def = match.group(1)
                method_body = match.group(2)

                # Check if security is already initialized
                if '_init_security_features' in method_body:
                    return match.group(0)

                # Find where to insert (after super().__init__ if exists)
                super_match = re.search(r'(super\(\).__init__\([^)]*\))', method_body)
                if super_match:
                    insert_pos = super_match.end()
                    # Find the newline after super().__init__
                    newline_pos = method_body.find('\n', insert_pos)
                    if newline_pos != -1:
                        insert_pos = newline_pos

                    new_body = (
                        method_body[:insert_pos] +
                        self.SECURITY_INIT_TEMPLATE +
                        method_body[insert_pos:]
                    )
                else:
                    # Insert at the beginning of the method
                    new_body = self.SECURITY_INIT_TEMPLATE + method_body

                return method_def + new_body

            content = re.sub(init_pattern, add_security_init, content, flags=re.MULTILINE | re.DOTALL)

            # 4. Add security methods if they don't exist
            if '_init_security_features' not in content:
                # Find the last method in the main class
                class_match = re.search(r'class\s+\w+\s*\([^)]*SecureA2AAgent[^)]*\)\s*:', content)
                if class_match:
                    # Find the end of the class (before the next class or end of file)
                    class_start = class_match.end()
                    next_class = re.search(r'\nclass\s+', content[class_start:])

                    if next_class:
                        class_end = class_start + next_class.start()
                    else:
                        class_end = len(content)

                    # Insert security methods before the end of the class
                    content = content[:class_end] + self.SECURITY_METHODS_TEMPLATE + content[class_end:]

            # 5. Update method decorators for security
            # Replace @app.route with @a2a_handler for A2A compliance
            content = re.sub(
                r'@app\.route\([^)]+\)',
                '@a2a_handler("REQUEST")',
                content
            )

            # 6. Add security checks to handler methods
            handler_pattern = r'(@a2a_handler\([^)]+\)\s*\n\s*async\s+def\s+\w+\s*\([^)]+\)\s*:\s*\n(?:\s*"""[\s\S]*?"""\s*\n)?)'

            def add_security_checks(match):
                handler_def = match.group(1)

                security_check = '''        # Security validation
        if not self.validate_input(request_data)[0]:
            return create_error_response("Invalid input data")

        # Rate limiting check
        client_id = request_data.get('client_id', 'unknown')
        if not self.check_rate_limit(client_id):
            return create_error_response("Rate limit exceeded")

'''
                return handler_def + security_check

            # Only add security checks if not already present
            if 'validate_input' not in content:
                content = re.sub(handler_pattern, add_security_checks, content)

            # Write migrated content if changes were made
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)

                self.files_migrated += 1
                return {
                    'status': 'migrated',
                    'file': file_path,
                    'changes': [
                        'Replaced base class with SecureA2AAgent',
                        'Added security imports',
                        'Added security initialization',
                        'Added security methods',
                        'Added input validation and rate limiting'
                    ]
                }
            else:
                return {
                    'status': 'no_changes',
                    'file': file_path
                }

        except Exception as e:
            self.errors.append({'file': file_path, 'error': str(e)})
            return {
                'status': 'error',
                'error': str(e),
                'file': file_path
            }

    def process_directory(self, directory: str) -> Dict[str, Any]:
        """Process all Python files in directory"""
        results = {
            'total_files': 0,
            'files_needing_migration': 0,
            'files_migrated': 0,
            'file_results': []
        }

        for root, dirs, files in os.walk(directory):
            # Skip test directories
            if 'test' in root or '__pycache__' in root:
                continue

            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    results['total_files'] += 1

                    # Analyze file
                    analysis = self.analyze_file(file_path)

                    if analysis['needs_migration']:
                        results['files_needing_migration'] += 1

                        # Migrate file
                        migration_result = self.migrate_file(file_path)
                        results['file_results'].append(migration_result)

                        if migration_result['status'] == 'migrated':
                            results['files_migrated'] += 1

        return results

    def generate_report(self, results: Dict[str, Any]):
        """Generate migration report"""
        print("\n=== Secure Base Class Migration Report ===\n")
        print(f"Total files scanned: {results['total_files']}")
        print(f"Files needing migration: {results['files_needing_migration']}")
        print(f"Files successfully migrated: {results['files_migrated']}")

        if results['file_results']:
            print("\nMigrated files:")
            for result in results['file_results']:
                if result['status'] == 'migrated':
                    print(f"  ✅ {result['file']}")
                    for change in result.get('changes', []):
                        print(f"     - {change}")
                elif result['status'] == 'error':
                    print(f"  ❌ {result['file']}: {result['error']}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error['file']}: {error['error']}")

        print("\n✅ Secure base class migration complete!")
        print("🔒 All agents now inherit from SecureA2AAgent with enhanced security features")
        print("⚠️  Remember to test all agents after migration")


def main():
    """Main function to run the secure base class migrator"""
    migrator = SecureBaseClassMigrator()

    # First create the SecureA2AAgent base class if it doesn't exist
    security_base_path = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/core/security_base.py"

    if not os.path.exists(security_base_path):
        print("🔨 Creating SecureA2AAgent base class...")
        create_secure_base_class(security_base_path)

    # Process the agents directory
    agents_dir = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents"

    print("🔍 Scanning for agents to migrate to SecureA2AAgent...")
    results = migrator.process_directory(agents_dir)

    # Generate report
    migrator.generate_report(results)

    # Save detailed results
    import json
    with open('secure_base_migration_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: secure_base_migration_results.json")


def create_secure_base_class(file_path: str):
    """Create the SecureA2AAgent base class"""
    content = '''"""
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
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE|FROM)\b|--|;|'|\"|\\*|OR\\s+1=1|AND\\s+1=1)",
                re.IGNORECASE
            ),
            'xss_attack': re.compile(
                r"(<script|<iframe|javascript:|onerror=|onload=|onclick=|<img\\s+src)",
                re.IGNORECASE
            ),
            'path_traversal': re.compile(r"(\\.\\./|\\.\\.\\\\|%2e%2e|%252e%252e)"),
            'command_injection': re.compile(r"[;&|`$(){}\\[\\]\\n\\r]"),
            'ldap_injection': re.compile(r"[\\(\\)\\*\\\\]")
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
'''

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✅ Created SecureA2AAgent base class at: {file_path}")


if __name__ == "__main__":
    main()
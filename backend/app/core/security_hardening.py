"""
Security Hardening Framework for A2A Agents
Provides authentication, authorization, secrets management, and vulnerability scanning
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from typing import Dict, Any, List, Optional, Set, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..a2a.sdk.types import A2AMessage
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(str, Enum):
    """Authentication methods"""
    JWT_TOKEN = "jwt_token"
    API_KEY = "api_key"
    MUTUAL_TLS = "mutual_tls"
    OAUTH2 = "oauth2"
    AGENT_CERTIFICATE = "agent_certificate"


class Permission(str, Enum):
    """System permissions"""
    READ_MESSAGES = "read_messages"
    WRITE_MESSAGES = "write_messages"
    EXECUTE_TASKS = "execute_tasks"
    MANAGE_AGENTS = "manage_agents"
    VIEW_METRICS = "view_metrics"
    ADMIN_ACCESS = "admin_access"
    SYSTEM_CONFIG = "system_config"


class VulnerabilityLevel(str, Enum):
    """Vulnerability severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    name: str
    description: str
    security_level: SecurityLevel
    required_auth_methods: List[AuthenticationMethod]
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 100
    audit_all_actions: bool = True


@dataclass
class Principal:
    """Security principal (user, agent, service)"""
    principal_id: str
    principal_type: str
    name: str
    permissions: Set[Permission]
    authentication_methods: List[AuthenticationMethod]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityToken:
    """Security token for authentication"""
    token_id: str
    principal_id: str
    token_type: str
    token_value: str
    expires_at: datetime
    permissions: Set[Permission]
    issued_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Security audit event"""
    event_id: str
    timestamp: datetime
    principal_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityVulnerability:
    """Security vulnerability report"""
    vulnerability_id: str
    title: str
    description: str
    severity: VulnerabilityLevel
    affected_component: str
    discovery_date: datetime
    remediation_status: str = "open"
    remediation_notes: str = ""
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None


class SecretManager:
    """Secure secret management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.fernet = Fernet(master_key)
        else:
            # Generate key from password
            password = os.environ.get('A2A_MASTER_PASSWORD', 'default-dev-password').encode()
            salt = os.environ.get('A2A_SALT', 'default-salt').encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.fernet = Fernet(key)
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret"""
        try:
            encrypted = self.fernet.encrypt(secret.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt secret: {e}")
            raise
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_secret.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
    
    def generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(length)
    
    def generate_password(self, length: int = 16) -> str:
        """Generate a secure password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class JWTTokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self,
        principal_id: str,
        permissions: Set[Permission],
        expires_in_minutes: int = 60,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """Create a JWT token"""
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=expires_in_minutes)
        
        payload = {
            'sub': principal_id,  # Subject
            'iat': int(now.timestamp()),  # Issued at
            'exp': int(expires_at.timestamp()),  # Expires
            'permissions': [p.value for p in permissions],
            'token_type': 'access_token'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError as e:
            raise Exception(f"Invalid token: {str(e)}")
    
    def refresh_token(self, token: str, extends_minutes: int = 60) -> str:
        """Refresh a JWT token"""
        try:
            payload = self.verify_token(token)
            
            # Create new token with extended expiration
            new_token = self.create_token(
                principal_id=payload['sub'],
                permissions=set(Permission(p) for p in payload['permissions']),
                expires_in_minutes=extends_minutes
            )
            
            return new_token
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise


class AuthenticationManager:
    """Manages authentication for A2A agents"""
    
    def __init__(self, redis_config: RedisConfig = None):
        self.redis_client = RedisClient(redis_config or RedisConfig())
        self.secret_manager = SecretManager()
        self.jwt_manager = JWTTokenManager(
            secret_key=os.environ.get('A2A_JWT_SECRET', 'dev-secret-key')
        )
        
        self.principals: Dict[str, Principal] = {}
        self.active_tokens: Dict[str, SecurityToken] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
    
    async def initialize(self):
        """Initialize authentication manager"""
        await self.redis_client.initialize()
        
        # Load principals from storage
        await self._load_principals()
        
        logger.info("Authentication manager initialized")
    
    async def shutdown(self):
        """Shutdown authentication manager"""
        await self.redis_client.close()
        logger.info("Authentication manager shut down")
    
    async def register_principal(
        self,
        principal_id: str,
        principal_type: str,
        name: str,
        permissions: Set[Permission],
        authentication_methods: List[AuthenticationMethod],
        password: Optional[str] = None
    ) -> Principal:
        """Register a new principal"""
        
        if principal_id in self.principals:
            raise Exception(f"Principal {principal_id} already exists")
        
        principal = Principal(
            principal_id=principal_id,
            principal_type=principal_type,
            name=name,
            permissions=permissions,
            authentication_methods=authentication_methods
        )
        
        # Store password hash if provided
        if password:
            password_hash = self.secret_manager.hash_password(password)
            principal.metadata['password_hash'] = password_hash
        
        # Generate API key if required
        if AuthenticationMethod.API_KEY in authentication_methods:
            api_key = self.secret_manager.generate_api_key()
            principal.metadata['api_key'] = api_key
            logger.info(f"Generated API key for principal {principal_id}: {api_key}")
        
        self.principals[principal_id] = principal
        
        # Store in Redis
        await self._store_principal(principal)
        
        logger.info(f"Registered principal: {principal_id} ({principal_type})")
        return principal
    
    @trace_async("authenticate")
    async def authenticate(
        self,
        principal_id: str,
        credentials: Dict[str, Any],
        auth_method: AuthenticationMethod,
        ip_address: Optional[str] = None
    ) -> Optional[SecurityToken]:
        """Authenticate a principal"""
        
        add_span_attributes({
            "auth.principal_id": principal_id,
            "auth.method": auth_method.value,
            "auth.ip_address": ip_address or "unknown"
        })
        
        # Check if account is locked
        if await self._is_account_locked(principal_id):
            await self._audit_event(principal_id, "authentication", "account", "locked", ip_address)
            raise Exception("Account is locked due to too many failed attempts")
        
        # Get principal
        principal = self.principals.get(principal_id)
        if not principal or not principal.is_active:
            await self._record_failed_attempt(principal_id)
            await self._audit_event(principal_id, "authentication", "account", "denied", ip_address)
            raise Exception("Authentication failed")
        
        # Check if authentication method is allowed
        if auth_method not in principal.authentication_methods:
            await self._audit_event(principal_id, "authentication", "method", "denied", ip_address)
            raise Exception(f"Authentication method {auth_method} not allowed")
        
        # Authenticate based on method
        try:
            if auth_method == AuthenticationMethod.JWT_TOKEN:
                await self._authenticate_jwt(principal, credentials)
            elif auth_method == AuthenticationMethod.API_KEY:
                await self._authenticate_api_key(principal, credentials)
            elif auth_method == AuthenticationMethod.OAUTH2:
                await self._authenticate_oauth2(principal, credentials)
            else:
                raise Exception(f"Unsupported authentication method: {auth_method}")
            
            # Create security token
            token = await self._create_security_token(principal)
            
            # Update last login
            principal.last_login = datetime.utcnow()
            await self._store_principal(principal)
            
            # Clear failed attempts
            if principal_id in self.failed_attempts:
                del self.failed_attempts[principal_id]
            
            await self._audit_event(principal_id, "authentication", "login", "success", ip_address)
            
            logger.info(f"Successful authentication for {principal_id}")
            return token
            
        except Exception as e:
            await self._record_failed_attempt(principal_id)
            await self._audit_event(principal_id, "authentication", "login", "failure", ip_address)
            logger.warning(f"Authentication failed for {principal_id}: {str(e)}")
            raise e
    
    async def _authenticate_jwt(self, principal: Principal, credentials: Dict[str, Any]):
        """Authenticate using JWT token"""
        token = credentials.get('token')
        if not token:
            raise Exception("JWT token required")
        
        payload = self.jwt_manager.verify_token(token)
        if payload['sub'] != principal.principal_id:
            raise Exception("Token principal mismatch")
    
    async def _authenticate_api_key(self, principal: Principal, credentials: Dict[str, Any]):
        """Authenticate using API key"""
        provided_key = credentials.get('api_key')
        if not provided_key:
            raise Exception("API key required")
        
        stored_key = principal.metadata.get('api_key')
        if not stored_key or provided_key != stored_key:
            raise Exception("Invalid API key")
    
    async def _authenticate_oauth2(self, principal: Principal, credentials: Dict[str, Any]):
        """Authenticate using OAuth2"""
        # Simplified OAuth2 validation
        access_token = credentials.get('access_token')
        if not access_token:
            raise Exception("OAuth2 access token required")
        
        # In production, validate with OAuth2 provider
        # For now, accept any token that starts with 'oauth_'
        if not access_token.startswith('oauth_'):
            raise Exception("Invalid OAuth2 token format")
    
    async def _create_security_token(self, principal: Principal) -> SecurityToken:
        """Create a security token"""
        token_id = secrets.token_urlsafe(16)
        
        # Create JWT token
        jwt_token = self.jwt_manager.create_token(
            principal_id=principal.principal_id,
            permissions=principal.permissions,
            expires_in_minutes=60
        )
        
        token = SecurityToken(
            token_id=token_id,
            principal_id=principal.principal_id,
            token_type="bearer",
            token_value=jwt_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            permissions=principal.permissions
        )
        
        self.active_tokens[token_id] = token
        
        # Store in Redis with expiration
        await self.redis_client.setex(
            f"token:{token_id}",
            timedelta(hours=1),
            json.dumps({
                "token_id": token_id,
                "principal_id": principal.principal_id,
                "permissions": [p.value for p in principal.permissions],
                "expires_at": token.expires_at.isoformat()
            })
        )
        
        return token
    
    async def verify_token(self, token_value: str) -> Optional[SecurityToken]:
        """Verify a security token"""
        try:
            # Decode JWT to get token ID
            payload = self.jwt_manager.verify_token(token_value)
            
            # Find matching token
            for token in self.active_tokens.values():
                if (token.token_value == token_value and 
                    token.principal_id == payload['sub'] and
                    token.expires_at > datetime.utcnow()):
                    return token
            
            return None
            
        except Exception:
            return None
    
    async def authorize_action(
        self,
        token: SecurityToken,
        required_permission: Permission,
        resource: str = ""
    ) -> bool:
        """Authorize an action"""
        
        # Check if token has required permission
        if required_permission not in token.permissions:
            await self._audit_event(
                token.principal_id,
                "authorization",
                resource or "unknown",
                "denied"
            )
            return False
        
        await self._audit_event(
            token.principal_id,
            "authorization",
            resource or "unknown",
            "granted"
        )
        return True
    
    async def _record_failed_attempt(self, principal_id: str):
        """Record a failed authentication attempt"""
        now = datetime.utcnow()
        
        if principal_id not in self.failed_attempts:
            self.failed_attempts[principal_id] = []
        
        self.failed_attempts[principal_id].append(now)
        
        # Keep only attempts from last 30 minutes
        cutoff = now - timedelta(minutes=30)
        self.failed_attempts[principal_id] = [
            attempt for attempt in self.failed_attempts[principal_id]
            if attempt > cutoff
        ]
        
        # Check if account should be locked
        if len(self.failed_attempts[principal_id]) >= 5:
            self.locked_accounts[principal_id] = now + timedelta(minutes=30)
            logger.warning(f"Account locked: {principal_id}")
    
    async def _is_account_locked(self, principal_id: str) -> bool:
        """Check if account is locked"""
        if principal_id not in self.locked_accounts:
            return False
        
        unlock_time = self.locked_accounts[principal_id]
        if datetime.utcnow() > unlock_time:
            del self.locked_accounts[principal_id]
            return False
        
        return True
    
    async def _audit_event(
        self,
        principal_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: Optional[str] = None
    ):
        """Record an audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.utcnow(),
            principal_id=principal_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address
        )
        
        # Store in Redis
        try:
            await self.redis_client.lpush(
                "audit_events",
                json.dumps({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "principal_id": event.principal_id,
                    "action": event.action,
                    "resource": event.resource,
                    "result": event.result,
                    "ip_address": event.ip_address
                })
            )
            
            # Keep only last 10000 events
            await self.redis_client.ltrim("audit_events", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
    
    async def _store_principal(self, principal: Principal):
        """Store principal in Redis"""
        try:
            data = {
                "principal_id": principal.principal_id,
                "principal_type": principal.principal_type,
                "name": principal.name,
                "permissions": [p.value for p in principal.permissions],
                "authentication_methods": [m.value for m in principal.authentication_methods],
                "created_at": principal.created_at.isoformat(),
                "last_login": principal.last_login.isoformat() if principal.last_login else None,
                "is_active": principal.is_active,
                "metadata": principal.metadata
            }
            
            await self.redis_client.hset(
                "principals",
                principal.principal_id,
                json.dumps(data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store principal: {e}")
    
    async def _load_principals(self):
        """Load principals from Redis"""
        try:
            principals_data = await self.redis_client.hgetall("principals")
            
            for principal_id, data_json in principals_data.items():
                data = json.loads(data_json)
                
                principal = Principal(
                    principal_id=data["principal_id"],
                    principal_type=data["principal_type"],
                    name=data["name"],
                    permissions=set(Permission(p) for p in data["permissions"]),
                    authentication_methods=[AuthenticationMethod(m) for m in data["authentication_methods"]],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_login=datetime.fromisoformat(data["last_login"]) if data["last_login"] else None,
                    is_active=data["is_active"],
                    metadata=data["metadata"]
                )
                
                self.principals[principal_id] = principal
            
            logger.info(f"Loaded {len(self.principals)} principals")
            
        except Exception as e:
            logger.error(f"Failed to load principals: {e}")


class VulnerabilityScanner:
    """Security vulnerability scanner"""
    
    def __init__(self):
        self.vulnerabilities: List[SecurityVulnerability] = []
        self.scan_rules: List[Callable] = []
        
        # Register default scan rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default vulnerability scan rules"""
        self.scan_rules.extend([
            self._scan_weak_passwords,
            self._scan_exposed_secrets,
            self._scan_insecure_permissions,
            self._scan_outdated_dependencies,
            self._scan_weak_encryption
        ])
    
    async def scan_system(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Perform comprehensive security scan"""
        vulnerabilities = []
        
        logger.info("Starting security vulnerability scan")
        
        for rule in self.scan_rules:
            try:
                rule_vulnerabilities = await rule(context)
                vulnerabilities.extend(rule_vulnerabilities)
            except Exception as e:
                logger.error(f"Scan rule failed: {rule.__name__}: {e}")
        
        self.vulnerabilities = vulnerabilities
        
        # Categorize by severity
        severity_counts = {}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] = severity_counts.get(vuln.severity.value, 0) + 1
        
        logger.info(f"Security scan completed. Found {len(vulnerabilities)} vulnerabilities: {severity_counts}")
        
        return vulnerabilities
    
    async def _scan_weak_passwords(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Scan for weak passwords"""
        vulnerabilities = []
        
        # Check for default passwords
        default_passwords = ['password', '123456', 'admin', 'default']
        
        for password in default_passwords:
            if password in str(context):
                vulnerabilities.append(SecurityVulnerability(
                    vulnerability_id=f"weak_password_{int(time.time())}",
                    title="Weak Password Detected",
                    description=f"Default or weak password '{password}' detected in system",
                    severity=VulnerabilityLevel.HIGH,
                    affected_component="authentication",
                    discovery_date=datetime.utcnow()
                ))
        
        return vulnerabilities
    
    async def _scan_exposed_secrets(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Scan for exposed secrets"""
        vulnerabilities = []
        
        # Common secret patterns
        secret_patterns = [
            r'api[_-]?key',
            r'secret[_-]?key',
            r'password',
            r'token',
            r'private[_-]?key'
        ]
        
        import re
        
        context_str = str(context).lower()
        for pattern in secret_patterns:
            if re.search(pattern, context_str):
                vulnerabilities.append(SecurityVulnerability(
                    vulnerability_id=f"exposed_secret_{int(time.time())}",
                    title="Potentially Exposed Secret",
                    description=f"Potential secret matching pattern '{pattern}' found in context",
                    severity=VulnerabilityLevel.MEDIUM,
                    affected_component="secrets_management",
                    discovery_date=datetime.utcnow()
                ))
        
        return vulnerabilities
    
    async def _scan_insecure_permissions(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Scan for insecure permissions"""
        vulnerabilities = []
        
        # Check for overly permissive configurations
        permissions = context.get('permissions', {})
        
        if permissions.get('allow_all', False):
            vulnerabilities.append(SecurityVulnerability(
                vulnerability_id=f"insecure_permissions_{int(time.time())}",
                title="Overly Permissive Configuration",
                description="System configured with 'allow_all' permissions",
                severity=VulnerabilityLevel.HIGH,
                affected_component="authorization",
                discovery_date=datetime.utcnow()
            ))
        
        return vulnerabilities
    
    async def _scan_outdated_dependencies(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Scan for outdated dependencies"""
        vulnerabilities = []
        
        # This would integrate with actual dependency scanners
        # For demonstration, we'll simulate finding outdated packages
        
        dependencies = context.get('dependencies', [])
        for dep in dependencies:
            if 'vulnerable' in dep.lower():
                vulnerabilities.append(SecurityVulnerability(
                    vulnerability_id=f"outdated_dep_{int(time.time())}",
                    title="Outdated Dependency",
                    description=f"Potentially vulnerable dependency: {dep}",
                    severity=VulnerabilityLevel.MEDIUM,
                    affected_component="dependencies",
                    discovery_date=datetime.utcnow()
                ))
        
        return vulnerabilities
    
    async def _scan_weak_encryption(self, context: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Scan for weak encryption"""
        vulnerabilities = []
        
        encryption_config = context.get('encryption', {})
        
        # Check for weak algorithms
        weak_algorithms = ['md5', 'sha1', 'des', 'rc4']
        
        for algorithm in weak_algorithms:
            if algorithm in str(encryption_config).lower():
                vulnerabilities.append(SecurityVulnerability(
                    vulnerability_id=f"weak_crypto_{int(time.time())}",
                    title="Weak Cryptographic Algorithm",
                    description=f"Weak encryption algorithm detected: {algorithm}",
                    severity=VulnerabilityLevel.HIGH,
                    affected_component="cryptography",
                    discovery_date=datetime.utcnow()
                ))
        
        return vulnerabilities


class SecurityHardeningFramework:
    """Main security hardening framework"""
    
    def __init__(self, redis_config: RedisConfig = None):
        self.redis_config = redis_config or RedisConfig()
        self.auth_manager = AuthenticationManager(self.redis_config)
        self.secret_manager = SecretManager()
        self.vulnerability_scanner = VulnerabilityScanner()
        
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.active_scans: Dict[str, datetime] = {}
        
    async def initialize(self):
        """Initialize security framework"""
        await self.auth_manager.initialize()
        
        # Create default security policies
        await self._create_default_policies()
        
        # Register default principals
        await self._create_default_principals()
        
        logger.info("Security hardening framework initialized")
    
    async def shutdown(self):
        """Shutdown security framework"""
        await self.auth_manager.shutdown()
        logger.info("Security hardening framework shut down")
    
    async def _create_default_policies(self):
        """Create default security policies"""
        
        # High security policy for production
        high_security = SecurityPolicy(
            name="high_security",
            description="High security policy for production environments",
            security_level=SecurityLevel.HIGH,
            required_auth_methods=[AuthenticationMethod.JWT_TOKEN, AuthenticationMethod.MUTUAL_TLS],
            session_timeout_minutes=30,
            max_failed_attempts=3,
            lockout_duration_minutes=60,
            require_mfa=True,
            rate_limit_requests_per_minute=50,
            audit_all_actions=True
        )
        
        self.security_policies["high_security"] = high_security
        
        # Development policy
        dev_security = SecurityPolicy(
            name="development",
            description="Relaxed security policy for development",
            security_level=SecurityLevel.MEDIUM,
            required_auth_methods=[AuthenticationMethod.API_KEY],
            session_timeout_minutes=120,
            max_failed_attempts=10,
            lockout_duration_minutes=10,
            require_mfa=False,
            rate_limit_requests_per_minute=200,
            audit_all_actions=False
        )
        
        self.security_policies["development"] = dev_security
    
    async def _create_default_principals(self):
        """Create default system principals"""
        
        # System admin
        await self.auth_manager.register_principal(
            principal_id="system_admin",
            principal_type="admin",
            name="System Administrator",
            permissions={
                Permission.READ_MESSAGES,
                Permission.WRITE_MESSAGES,
                Permission.EXECUTE_TASKS,
                Permission.MANAGE_AGENTS,
                Permission.VIEW_METRICS,
                Permission.ADMIN_ACCESS,
                Permission.SYSTEM_CONFIG
            },
            authentication_methods=[AuthenticationMethod.JWT_TOKEN, AuthenticationMethod.API_KEY],
            password = os.getenv("PASSWORD", "")
        )
        
        # Agent service account
        await self.auth_manager.register_principal(
            principal_id="agent_service",
            principal_type="service",
            name="Agent Service Account",
            permissions={
                Permission.READ_MESSAGES,
                Permission.WRITE_MESSAGES,
                Permission.EXECUTE_TASKS,
                Permission.VIEW_METRICS
            },
            authentication_methods=[AuthenticationMethod.API_KEY, AuthenticationMethod.AGENT_CERTIFICATE]
        )
    
    async def perform_security_scan(self) -> List[SecurityVulnerability]:
        """Perform comprehensive security scan"""
        scan_id = secrets.token_urlsafe(8)
        self.active_scans[scan_id] = datetime.utcnow()
        
        try:
            # Gather system context for scanning
            context = {
                'principals': len(self.auth_manager.principals),
                'active_tokens': len(self.auth_manager.active_tokens),
                'policies': len(self.security_policies),
                'permissions': [p.value for p in Permission],
                'dependencies': ['requests==2.28.1', 'jwt==2.6.0', 'bcrypt==4.0.1'],  # Example
                'encryption': {'algorithm': 'AES-256', 'mode': 'GCM'}
            }
            
            vulnerabilities = await self.vulnerability_scanner.scan_system(context)
            
            return vulnerabilities
            
        finally:
            if scan_id in self.active_scans:
                del self.active_scans[scan_id]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status"""
        return {
            "auth_manager": {
                "principals": len(self.auth_manager.principals),
                "active_tokens": len(self.auth_manager.active_tokens),
                "locked_accounts": len(self.auth_manager.locked_accounts)
            },
            "policies": len(self.security_policies),
            "vulnerabilities": len(self.vulnerability_scanner.vulnerabilities),
            "active_scans": len(self.active_scans),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global security framework
_security_framework = None


async def initialize_security_hardening(redis_config: RedisConfig = None) -> SecurityHardeningFramework:
    """Initialize global security hardening framework"""
    global _security_framework
    
    if _security_framework is None:
        _security_framework = SecurityHardeningFramework(redis_config)
        await _security_framework.initialize()
    
    return _security_framework


async def get_security_framework() -> Optional[SecurityHardeningFramework]:
    """Get the global security framework"""
    return _security_framework


async def shutdown_security_hardening():
    """Shutdown global security hardening framework"""
    global _security_framework
    
    if _security_framework:
        await _security_framework.shutdown()
        _security_framework = None


# Security decorators
def require_permission(permission: Permission):
    """Decorator to require specific permission for function access"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Extract token from request context
            # This would be implemented based on your request handling
            token = getattr(self, '_current_token', None)
            
            if not token:
                raise Exception("Authentication required")
            
            framework = await get_security_framework()
            if not framework:
                raise Exception("Security framework not initialized")
            
            authorized = await framework.auth_manager.authorize_action(
                token, permission, func.__name__
            )
            
            if not authorized:
                raise Exception(f"Permission denied: {permission.value}")
            
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


def audit_action(action: str, resource: str = ""):
    """Decorator to audit function calls"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            token = getattr(self, '_current_token', None)
            principal_id = token.principal_id if token else "unknown"
            
            try:
                result = await func(self, *args, **kwargs)
                
                framework = await get_security_framework()
                if framework:
                    await framework.auth_manager._audit_event(
                        principal_id, action, resource or func.__name__, "success"
                    )
                
                return result
                
            except Exception as e:
                framework = await get_security_framework()
                if framework:
                    await framework.auth_manager._audit_event(
                        principal_id, action, resource or func.__name__, "failure"
                    )
                raise e
        return wrapper
    return decorator
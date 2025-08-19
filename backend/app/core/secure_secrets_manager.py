"""
Secure Secrets Management for A2A Agents
Provides enterprise-grade secrets management with encryption, rotation, and audit logging
"""

import asyncio
import json
import logging
import os
import secrets
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import base64
from pathlib import Path

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import jwt

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    PRIVATE_KEY = "private_key"
    CERTIFICATE = "certificate"
    OAUTH_TOKEN = "oauth_token"
    WEBHOOK_SECRET = "webhook_secret"


class SecretScope(str, Enum):
    """Scope of secret access"""
    GLOBAL = "global"
    SERVICE = "service"
    AGENT = "agent"
    USER = "user"
    TEMPORARY = "temporary"


class RotationPolicy(str, Enum):
    """Secret rotation policies"""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_COMPROMISE = "on_compromise"


@dataclass
class SecretMetadata:
    """Metadata for a managed secret"""
    secret_id: str
    secret_type: SecretType
    scope: SecretScope
    created_at: datetime
    last_rotated: datetime
    rotation_policy: RotationPolicy
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    allowed_principals: Set[str] = field(default_factory=set)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SecretVersion:
    """A version of a secret value"""
    version_id: str
    secret_id: str
    encrypted_value: bytes
    created_at: datetime
    is_current: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecretProvider(ABC):
    """Abstract base class for secret storage providers"""
    
    @abstractmethod
    async def store_secret(self, secret_id: str, encrypted_value: bytes, metadata: SecretMetadata) -> bool:
        """Store an encrypted secret"""
        pass
    
    @abstractmethod
    async def retrieve_secret(self, secret_id: str, version_id: Optional[str] = None) -> Optional[bytes]:
        """Retrieve an encrypted secret"""
        pass
    
    @abstractmethod
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret"""
        pass
    
    @abstractmethod
    async def list_secrets(self, scope: Optional[SecretScope] = None) -> List[str]:
        """List secret IDs"""
        pass


class RedisSecretProvider(SecretProvider):
    """Redis-based secret storage provider"""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis_client = RedisClient(redis_config)
        self.prefix = "a2a:secrets:"
        
    async def initialize(self):
        """Initialize the provider"""
        await self.redis_client.initialize()
        
    async def shutdown(self):
        """Shutdown the provider"""
        await self.redis_client.close()
    
    async def store_secret(self, secret_id: str, encrypted_value: bytes, metadata: SecretMetadata) -> bool:
        """Store an encrypted secret in Redis"""
        try:
            # Create version entry
            version_id = f"{secret_id}:{int(time.time())}"
            version_data = {
                "version_id": version_id,
                "secret_id": secret_id,
                "encrypted_value": base64.b64encode(encrypted_value).decode(),
                "created_at": metadata.created_at.isoformat(),
                "is_current": True
            }
            
            # Store version
            await self.redis_client.hset(
                f"{self.prefix}versions",
                version_id,
                json.dumps(version_data)
            )
            
            # Store metadata
            metadata_data = {
                "secret_id": metadata.secret_id,
                "secret_type": metadata.secret_type.value,
                "scope": metadata.scope.value,
                "created_at": metadata.created_at.isoformat(),
                "last_rotated": metadata.last_rotated.isoformat(),
                "rotation_policy": metadata.rotation_policy.value,
                "access_count": metadata.access_count,
                "last_accessed": metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "tags": metadata.tags,
                "allowed_principals": list(metadata.allowed_principals),
                "current_version": version_id
            }
            
            await self.redis_client.hset(
                f"{self.prefix}metadata",
                secret_id,
                json.dumps(metadata_data)
            )
            
            # Mark previous versions as non-current
            await self._mark_previous_versions_old(secret_id, version_id)
            
            # Set expiration if specified
            if metadata.expires_at:
                ttl = int((metadata.expires_at - datetime.utcnow()).total_seconds())
                await self.redis_client.expire(f"{self.prefix}versions:{version_id}", ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            return False
    
    async def retrieve_secret(self, secret_id: str, version_id: Optional[str] = None) -> Optional[bytes]:
        """Retrieve an encrypted secret from Redis"""
        try:
            if not version_id:
                # Get current version
                metadata_json = await self.redis_client.hget(f"{self.prefix}metadata", secret_id)
                if not metadata_json:
                    return None
                
                metadata = json.loads(metadata_json)
                version_id = metadata.get("current_version")
                
                if not version_id:
                    return None
            
            # Get version data
            version_json = await self.redis_client.hget(f"{self.prefix}versions", version_id)
            if not version_json:
                return None
            
            version_data = json.loads(version_json)
            encrypted_value = base64.b64decode(version_data["encrypted_value"])
            
            # Update access metadata
            await self._update_access_metadata(secret_id)
            
            return encrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            return None
    
    async def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret and all its versions"""
        try:
            # Get all versions for this secret
            all_versions = await self.redis_client.hgetall(f"{self.prefix}versions")
            secret_versions = [
                version_id for version_id, data in all_versions.items()
                if json.loads(data)["secret_id"] == secret_id
            ]
            
            # Delete all versions
            if secret_versions:
                await self.redis_client.hdel(f"{self.prefix}versions", *secret_versions)
            
            # Delete metadata
            await self.redis_client.hdel(f"{self.prefix}metadata", secret_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            return False
    
    async def list_secrets(self, scope: Optional[SecretScope] = None) -> List[str]:
        """List secret IDs, optionally filtered by scope"""
        try:
            all_metadata = await self.redis_client.hgetall(f"{self.prefix}metadata")
            
            if not scope:
                return list(all_metadata.keys())
            
            filtered_secrets = []
            for secret_id, metadata_json in all_metadata.items():
                metadata = json.loads(metadata_json)
                if metadata.get("scope") == scope.value:
                    filtered_secrets.append(secret_id)
            
            return filtered_secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def _mark_previous_versions_old(self, secret_id: str, current_version_id: str):
        """Mark all previous versions as non-current"""
        try:
            all_versions = await self.redis_client.hgetall(f"{self.prefix}versions")
            
            for version_id, version_json in all_versions.items():
                version_data = json.loads(version_json)
                if (version_data["secret_id"] == secret_id and 
                    version_id != current_version_id):
                    version_data["is_current"] = False
                    await self.redis_client.hset(
                        f"{self.prefix}versions",
                        version_id,
                        json.dumps(version_data)
                    )
                    
        except Exception as e:
            logger.error(f"Failed to mark previous versions as old: {e}")
    
    async def _update_access_metadata(self, secret_id: str):
        """Update access count and timestamp"""
        try:
            metadata_json = await self.redis_client.hget(f"{self.prefix}metadata", secret_id)
            if metadata_json:
                metadata = json.loads(metadata_json)
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = datetime.utcnow().isoformat()
                
                await self.redis_client.hset(
                    f"{self.prefix}metadata",
                    secret_id,
                    json.dumps(metadata)
                )
                
        except Exception as e:
            logger.error(f"Failed to update access metadata: {e}")


class EncryptionManager:
    """Handles encryption/decryption of secrets"""
    
    def __init__(self, master_key_path: Optional[str] = None):
        self.master_key_path = master_key_path or os.path.join(os.getcwd(), ".secrets", "master.key")
        self.master_key = self._load_or_generate_master_key()
        self.fernet = Fernet(self.master_key)
        
        # RSA key pair for asymmetric encryption
        self.private_key = self._load_or_generate_rsa_key()
        self.public_key = self.private_key.public_key()
    
    def _load_or_generate_master_key(self) -> bytes:
        """Load existing master key or generate new one"""
        master_key_file = Path(self.master_key_path)
        master_key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if master_key_file.exists():
            try:
                with open(master_key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load master key: {e}, generating new one")
        
        # Generate new master key
        master_key = Fernet.generate_key()
        
        try:
            # Set restrictive permissions before writing
            master_key_file.touch(mode=0o600)
            with open(master_key_file, 'wb') as f:
                f.write(master_key)
            logger.info(f"Generated new master key at {master_key_file}")
        except Exception as e:
            logger.error(f"Failed to save master key: {e}")
            
        return master_key
    
    def _load_or_generate_rsa_key(self) -> rsa.RSAPrivateKey:
        """Load existing RSA key or generate new one"""
        rsa_key_path = self.master_key_path.replace("master.key", "rsa_private.pem")
        rsa_key_file = Path(rsa_key_path)
        
        if rsa_key_file.exists():
            try:
                with open(rsa_key_file, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None
                    )
                return private_key
            except Exception as e:
                logger.warning(f"Failed to load RSA key: {e}, generating new one")
        
        # Generate new RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        try:
            # Save private key
            rsa_key_file.touch(mode=0o600)
            with open(rsa_key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            pub_key_path = rsa_key_path.replace("_private.pem", "_public.pem")
            with open(pub_key_path, 'wb') as f:
                f.write(private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
                
            logger.info(f"Generated new RSA key pair at {rsa_key_file}")
            
        except Exception as e:
            logger.error(f"Failed to save RSA keys: {e}")
            
        return private_key
    
    def encrypt_secret(self, plaintext: str, use_asymmetric: bool = False) -> bytes:
        """Encrypt a secret value"""
        try:
            if use_asymmetric:
                # Use RSA for asymmetric encryption (smaller values only)
                if len(plaintext.encode()) > 190:  # RSA 2048 limit
                    raise ValueError("Value too large for RSA encryption, use symmetric instead")
                
                encrypted = self.public_key.encrypt(
                    plaintext.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return encrypted
            else:
                # Use Fernet for symmetric encryption
                return self.fernet.encrypt(plaintext.encode())
                
        except Exception as e:
            logger.error(f"Failed to encrypt secret: {e}")
            raise
    
    def decrypt_secret(self, encrypted_data: bytes, use_asymmetric: bool = False) -> str:
        """Decrypt a secret value"""
        try:
            if use_asymmetric:
                decrypted = self.private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted.decode()
            else:
                return self.fernet.decrypt(encrypted_data).decode()
                
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
    
    def generate_secret(self, secret_type: SecretType, length: int = 32) -> str:
        """Generate a new secret value"""
        if secret_type == SecretType.API_KEY:
            return f"a2a_{secrets.token_urlsafe(length)}"
        elif secret_type == SecretType.JWT_SECRET:
            return secrets.token_urlsafe(64)  # Longer for JWT
        elif secret_type == SecretType.DATABASE_PASSWORD:
            # Strong password with mixed characters
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return ''.join(secrets.choice(alphabet) for _ in range(length))
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return secrets.token_hex(32)
        else:
            return secrets.token_urlsafe(length)


class SecureSecretsManager:
    """Main secrets management system"""
    
    def __init__(self, redis_config: RedisConfig = None, master_key_path: Optional[str] = None):
        self.provider = RedisSecretProvider(redis_config or RedisConfig())
        self.encryption_manager = EncryptionManager(master_key_path)
        self.secrets_cache: Dict[str, tuple] = {}  # secret_id -> (value, expiry)
        self.cache_ttl = timedelta(minutes=5)  # Cache secrets for 5 minutes
        
        # Rotation policies
        self.rotation_handlers: Dict[SecretType, Callable] = {}
        self.auto_rotation_enabled = True
        
    async def initialize(self):
        """Initialize the secrets manager"""
        await self.provider.initialize()
        
        # Register default rotation handlers
        self._register_default_rotation_handlers()
        
        # Start background rotation task
        if self.auto_rotation_enabled:
            asyncio.create_task(self._rotation_worker())
        
        logger.info("Secure secrets manager initialized")
    
    async def shutdown(self):
        """Shutdown the secrets manager"""
        await self.provider.shutdown()
        self.secrets_cache.clear()
        logger.info("Secure secrets manager shut down")
    
    @trace_async("store_secret")
    async def store_secret(
        self,
        secret_id: str,
        value: str,
        secret_type: SecretType,
        scope: SecretScope = SecretScope.SERVICE,
        rotation_policy: RotationPolicy = RotationPolicy.MONTHLY,
        expires_at: Optional[datetime] = None,
        allowed_principals: Optional[Set[str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """Store a secret securely"""
        
        add_span_attributes({
            "secret.id": secret_id,
            "secret.type": secret_type.value,
            "secret.scope": scope.value
        })
        
        try:
            # Encrypt the secret
            encrypted_value = self.encryption_manager.encrypt_secret(value)
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                scope=scope,
                created_at=datetime.utcnow(),
                last_rotated=datetime.utcnow(),
                rotation_policy=rotation_policy,
                expires_at=expires_at,
                allowed_principals=allowed_principals or set(),
                tags=tags or {}
            )
            
            # Store in provider
            success = await self.provider.store_secret(secret_id, encrypted_value, metadata)
            
            if success:
                # Clear from cache to force refresh
                self.secrets_cache.pop(secret_id, None)
                
                # Log audit event
                await self._audit_event(secret_id, "secret_stored", {
                    "secret_type": secret_type.value,
                    "scope": scope.value
                })
                
                logger.info(f"Successfully stored secret: {secret_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            return False
    
    @trace_async("retrieve_secret")
    async def retrieve_secret(
        self,
        secret_id: str,
        principal_id: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[str]:
        """Retrieve a secret value"""
        
        add_span_attributes({
            "secret.id": secret_id,
            "principal.id": principal_id or "unknown"
        })
        
        try:
            # Check cache first
            if use_cache and secret_id in self.secrets_cache:
                value, expiry = self.secrets_cache[secret_id]
                if datetime.utcnow() < expiry:
                    return value
                else:
                    # Cache expired, remove
                    del self.secrets_cache[secret_id]
            
            # Retrieve encrypted value
            encrypted_value = await self.provider.retrieve_secret(secret_id)
            if not encrypted_value:
                return None
            
            # Check permissions if principal specified
            if principal_id:
                metadata = await self._get_secret_metadata(secret_id)
                if metadata and metadata.allowed_principals:
                    if principal_id not in metadata.allowed_principals:
                        await self._audit_event(secret_id, "access_denied", {
                            "principal_id": principal_id,
                            "reason": "insufficient_permissions"
                        })
                        raise PermissionError(f"Principal {principal_id} not authorized for secret {secret_id}")
            
            # Decrypt the secret
            decrypted_value = self.encryption_manager.decrypt_secret(encrypted_value)
            
            # Cache the result
            if use_cache:
                self.secrets_cache[secret_id] = (decrypted_value, datetime.utcnow() + self.cache_ttl)
            
            # Log audit event
            await self._audit_event(secret_id, "secret_accessed", {
                "principal_id": principal_id
            })
            
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            await self._audit_event(secret_id, "access_failed", {
                "principal_id": principal_id,
                "error": str(e)
            })
            return None
    
    async def rotate_secret(self, secret_id: str, new_value: Optional[str] = None) -> bool:
        """Rotate a secret to a new value"""
        try:
            metadata = await self._get_secret_metadata(secret_id)
            if not metadata:
                logger.error(f"Cannot rotate secret {secret_id}: metadata not found")
                return False
            
            # Generate new value if not provided
            if not new_value:
                rotation_handler = self.rotation_handlers.get(metadata.secret_type)
                if rotation_handler:
                    new_value = await rotation_handler(secret_id, metadata)
                else:
                    new_value = self.encryption_manager.generate_secret(metadata.secret_type)
            
            # Store new version
            success = await self.store_secret(
                secret_id=secret_id,
                value=new_value,
                secret_type=metadata.secret_type,
                scope=metadata.scope,
                rotation_policy=metadata.rotation_policy,
                expires_at=metadata.expires_at,
                allowed_principals=metadata.allowed_principals,
                tags=metadata.tags
            )
            
            if success:
                await self._audit_event(secret_id, "secret_rotated", {
                    "secret_type": metadata.secret_type.value
                })
                logger.info(f"Successfully rotated secret: {secret_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")
            return False
    
    async def delete_secret(self, secret_id: str, principal_id: Optional[str] = None) -> bool:
        """Delete a secret"""
        try:
            # Clear from cache
            self.secrets_cache.pop(secret_id, None)
            
            # Delete from provider
            success = await self.provider.delete_secret(secret_id)
            
            if success:
                await self._audit_event(secret_id, "secret_deleted", {
                    "principal_id": principal_id
                })
                logger.info(f"Successfully deleted secret: {secret_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            return False
    
    async def list_secrets(
        self,
        scope: Optional[SecretScope] = None,
        secret_type: Optional[SecretType] = None
    ) -> List[Dict[str, Any]]:
        """List secrets with metadata"""
        try:
            secret_ids = await self.provider.list_secrets(scope)
            
            secrets_info = []
            for secret_id in secret_ids:
                metadata = await self._get_secret_metadata(secret_id)
                if metadata:
                    if secret_type and metadata.secret_type != secret_type:
                        continue
                        
                    secrets_info.append({
                        "secret_id": secret_id,
                        "secret_type": metadata.secret_type.value,
                        "scope": metadata.scope.value,
                        "created_at": metadata.created_at.isoformat(),
                        "last_rotated": metadata.last_rotated.isoformat(),
                        "rotation_policy": metadata.rotation_policy.value,
                        "access_count": metadata.access_count,
                        "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                        "tags": metadata.tags
                    })
            
            return secrets_info
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def register_rotation_handler(self, secret_type: SecretType, handler: Callable):
        """Register a custom rotation handler for a secret type"""
        self.rotation_handlers[secret_type] = handler
        logger.info(f"Registered rotation handler for {secret_type.value}")
    
    async def _get_secret_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret"""
        # This would be implemented by extending the provider interface
        # For now, return None to indicate metadata retrieval needs implementation
        return None
    
    def _register_default_rotation_handlers(self):
        """Register default rotation handlers"""
        async def rotate_api_key(secret_id: str, metadata: SecretMetadata) -> str:
            return self.encryption_manager.generate_secret(SecretType.API_KEY)
        
        async def rotate_jwt_secret(secret_id: str, metadata: SecretMetadata) -> str:
            return self.encryption_manager.generate_secret(SecretType.JWT_SECRET)
        
        async def rotate_database_password(secret_id: str, metadata: SecretMetadata) -> str:
            return self.encryption_manager.generate_secret(SecretType.DATABASE_PASSWORD, 24)
        
        self.rotation_handlers[SecretType.API_KEY] = rotate_api_key
        self.rotation_handlers[SecretType.JWT_SECRET] = rotate_jwt_secret
        self.rotation_handlers[SecretType.DATABASE_PASSWORD] = rotate_database_password
    
    async def _rotation_worker(self):
        """Background worker for automatic secret rotation"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Get all secrets that need rotation
                all_secrets = await self.list_secrets()
                
                for secret_info in all_secrets:
                    secret_id = secret_info["secret_id"]
                    rotation_policy = RotationPolicy(secret_info["rotation_policy"])
                    last_rotated = datetime.fromisoformat(secret_info["last_rotated"])
                    
                    should_rotate = False
                    
                    if rotation_policy == RotationPolicy.DAILY:
                        should_rotate = datetime.utcnow() - last_rotated > timedelta(days=1)
                    elif rotation_policy == RotationPolicy.WEEKLY:
                        should_rotate = datetime.utcnow() - last_rotated > timedelta(weeks=1)
                    elif rotation_policy == RotationPolicy.MONTHLY:
                        should_rotate = datetime.utcnow() - last_rotated > timedelta(days=30)
                    elif rotation_policy == RotationPolicy.QUARTERLY:
                        should_rotate = datetime.utcnow() - last_rotated > timedelta(days=90)
                    
                    if should_rotate:
                        logger.info(f"Auto-rotating secret: {secret_id}")
                        await self.rotate_secret(secret_id)
                
            except Exception as e:
                logger.error(f"Error in rotation worker: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _audit_event(self, secret_id: str, action: str, details: Dict[str, Any]):
        """Log an audit event"""
        audit_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "secret_id": secret_id,
            "action": action,
            "details": details
        }
        
        # Store audit log (implementation depends on audit storage choice)
        logger.info(f"Audit: {action} for secret {secret_id}", extra=audit_data)


# Global secrets manager instance
_secrets_manager = None


async def initialize_secrets_manager(
    redis_config: RedisConfig = None,
    master_key_path: Optional[str] = None
) -> SecureSecretsManager:
    """Initialize global secrets manager"""
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = SecureSecretsManager(redis_config, master_key_path)
        await _secrets_manager.initialize()
    
    return _secrets_manager


async def get_secrets_manager() -> Optional[SecureSecretsManager]:
    """Get the global secrets manager"""
    return _secrets_manager


async def shutdown_secrets_manager():
    """Shutdown global secrets manager"""
    global _secrets_manager
    
    if _secrets_manager:
        await _secrets_manager.shutdown()
        _secrets_manager = None


# Convenience functions
async def store_secret(secret_id: str, value: str, secret_type: SecretType, **kwargs) -> bool:
    """Store a secret using the global manager"""
    manager = await get_secrets_manager()
    if not manager:
        raise RuntimeError("Secrets manager not initialized")
    return await manager.store_secret(secret_id, value, secret_type, **kwargs)


async def get_secret(secret_id: str, principal_id: Optional[str] = None) -> Optional[str]:
    """Retrieve a secret using the global manager"""
    manager = await get_secrets_manager()
    if not manager:
        raise RuntimeError("Secrets manager not initialized")
    return await manager.retrieve_secret(secret_id, principal_id)


async def rotate_secret(secret_id: str, new_value: Optional[str] = None) -> bool:
    """Rotate a secret using the global manager"""
    manager = await get_secrets_manager()
    if not manager:
        raise RuntimeError("Secrets manager not initialized")
    return await manager.rotate_secret(secret_id, new_value)
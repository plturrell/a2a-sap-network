"""
Secure Environment Variable and Secrets Management
Provides encrypted storage and secure access to sensitive configuration
"""

import os
import logging
import base64
import json
import hashlib
from typing import Dict, Any, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel
from contextlib import contextmanager
import tempfile
import secrets

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found"""
    pass


class SecretValidationError(Exception):
    """Raised when a secret fails validation"""
    pass


class SecretConfig(BaseModel):
    """Configuration for secrets management"""
    encryption_enabled: bool = True
    key_derivation_iterations: int = 100000
    secrets_file_path: str = "./data/secrets.enc"
    key_file_path: str = "./data/.secret_key"
    validate_secrets: bool = True
    require_env_vars: bool = True


class SecretsManager:
    """Secure secrets and environment variable manager"""

    def __init__(self, config: SecretConfig = None):
        self.config = config or SecretConfig()
        self._encryption_key: Optional[bytes] = None
        self._secrets_cache: Dict[str, str] = {}
        self._validated_secrets: set = set()

        # Ensure directories exist
        Path(self.config.secrets_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.key_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize encryption if enabled
        if self.config.encryption_enabled:
            self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption system"""
        try:
            # Try to load existing key
            if Path(self.config.key_file_path).exists():
                with open(self.config.key_file_path, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()

                # Save key securely
                with open(self.config.key_file_path, 'wb') as f:
                    f.write(self._encryption_key)

                # Set restrictive permissions
                os.chmod(self.config.key_file_path, 0o600)
                logger.info("Generated new encryption key for secrets")

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.config.encryption_enabled = False

    def get_secret(self,
                   secret_name: str,
                   default: Optional[str] = None,
                   required: bool = True,
                   validate_func: Optional[callable] = None) -> Optional[str]:
        """
        Securely retrieve a secret value

        Args:
            secret_name: Name of the secret
            default: Default value if secret not found
            required: Whether the secret is required
            validate_func: Optional validation function

        Returns:
            Secret value or default

        Raises:
            SecretNotFoundError: If required secret is not found
            SecretValidationError: If secret fails validation
        """
        # Check cache first
        if secret_name in self._secrets_cache:
            value = self._secrets_cache[secret_name]
        else:
            # Try environment variable first
            value = os.getenv(secret_name)

            # If not in env, try encrypted storage
            if not value and self.config.encryption_enabled:
                value = self._load_encrypted_secret(secret_name)

            # Cache the value
            if value:
                self._secrets_cache[secret_name] = value

        # Use default if no value found
        if not value:
            value = default

        # Check if required
        if required and not value:
            raise SecretNotFoundError(f"Required secret '{secret_name}' not found")

        # Validate secret
        if value and validate_func and secret_name not in self._validated_secrets:
            try:
                if not validate_func(value):
                    raise SecretValidationError(f"Secret '{secret_name}' failed validation")
                self._validated_secrets.add(secret_name)
            except Exception as e:
                raise SecretValidationError(f"Secret '{secret_name}' validation error: {e}")

        return value

    def set_secret(self, secret_name: str, secret_value: str, encrypt: bool = True):
        """
        Securely store a secret value

        Args:
            secret_name: Name of the secret
            secret_value: Value to store
            encrypt: Whether to encrypt the value
        """
        # Update cache
        self._secrets_cache[secret_name] = secret_value

        # Store encrypted if enabled
        if encrypt and self.config.encryption_enabled:
            self._save_encrypted_secret(secret_name, secret_value)

        logger.info(f"Secret '{secret_name}' updated")

    def _load_encrypted_secret(self, secret_name: str) -> Optional[str]:
        """Load encrypted secret from file"""
        if not self.config.encryption_enabled or not self._encryption_key:
            return None

        try:
            secrets_file = Path(self.config.secrets_file_path)
            if not secrets_file.exists():
                return None

            # Load and decrypt secrets file
            with open(secrets_file, 'rb') as f:
                encrypted_data = f.read()

            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted_data.decode('utf-8'))

            return secrets_dict.get(secret_name)

        except Exception as e:
            logger.error(f"Failed to load encrypted secret '{secret_name}': {e}")
            return None

    def _save_encrypted_secret(self, secret_name: str, secret_value: str):
        """Save encrypted secret to file"""
        if not self.config.encryption_enabled or not self._encryption_key:
            return

        try:
            # Load existing secrets
            secrets_dict = {}
            secrets_file = Path(self.config.secrets_file_path)

            if secrets_file.exists():
                with open(secrets_file, 'rb') as f:
                    encrypted_data = f.read()

                fernet = Fernet(self._encryption_key)
                decrypted_data = fernet.decrypt(encrypted_data)
                secrets_dict = json.loads(decrypted_data.decode('utf-8'))

            # Update secret
            secrets_dict[secret_name] = secret_value

            # Encrypt and save
            fernet = Fernet(self._encryption_key)
            json_data = json.dumps(secrets_dict).encode('utf-8')
            encrypted_data = fernet.encrypt(json_data)

            # Atomic write using temporary file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                           dir=secrets_file.parent) as temp_file:
                temp_file.write(encrypted_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = temp_file.name

            # Move temporary file to final location
            os.replace(temp_path, secrets_file)
            os.chmod(secrets_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save encrypted secret '{secret_name}': {e}")

    def get_database_url(self,
                        db_type: str = "primary",
                        validate: bool = True) -> str:
        """Get database URL with validation"""

        env_var_name = f"{db_type.upper()}_DATABASE_URL"

        def validate_db_url(url: str) -> bool:
            """Validate database URL format and security"""
            if not url:
                return False

            # Check for basic URL format
            if not any(url.startswith(prefix) for prefix in
                      ['postgresql://', 'mysql://', 'sqlite:///', 'oracle://', 'hana://']):
                return False

            # Security checks for production databases
            if not url.startswith('sqlite:'):
                # Check for SSL/TLS requirement
                security_params = ['sslmode=require', 'ssl=true', 'encrypt=true']
                if not any(param in url.lower() for param in security_params):
                    logger.warning(f"⚠️ Database URL for {db_type} should include SSL/TLS")

                # Check for credentials in URL (should be avoided)
                if '@' in url and ':' in url.split('@')[0]:
                    logger.warning(f"⚠️ Database credentials in URL for {db_type} - consider using env vars")

            return True

        return self.get_secret(
            env_var_name,
            required=True,
            validate_func=validate_db_url if validate else None
        )

    def get_api_key(self,
                   service_name: str,
                   validate: bool = True) -> str:
        """Get API key with validation"""

        env_var_name = f"{service_name.upper()}_API_KEY"

        def validate_api_key(key: str) -> bool:
            """Validate API key format"""
            if not key:
                return False

            # Basic length check
            if len(key) < 16:
                return False

            # Check for common patterns
            if key.startswith('sk-') and len(key) < 20:  # OpenAI style
                return False

            if key.startswith('your-xai-api-key-here') and len(key) < 32:  # X.AI style
                return False

            return True

        return self.get_secret(
            env_var_name,
            required=True,
            validate_func=validate_api_key if validate else None
        )

    def get_jwt_secret(self) -> str:
        """Get JWT secret with strong validation"""

        def validate_jwt_secret(secret: str) -> bool:
            """Validate JWT secret strength"""
            if not secret or len(secret) < 32:
                return False

            # Check entropy - should have mix of characters
            has_lower = any(c.islower() for c in secret)
            has_upper = any(c.isupper() for c in secret)
            has_digit = any(c.isdigit() for c in secret)
            has_special = any(c in '!@#$%^&*()_+-=' for c in secret)

            return sum([has_lower, has_upper, has_digit, has_special]) >= 3

        return self.get_secret(
            "JWT_SECRET_KEY",
            required=True,
            validate_func=validate_jwt_secret
        )

    def validate_all_secrets(self) -> Dict[str, bool]:
        """Validate all configured secrets"""
        validation_results = {}

        # Essential secrets for A2A platform
        essential_secrets = {
            "DATABASE_URL": lambda x: x and any(x.startswith(p) for p in ['postgresql://', 'sqlite:///']),
            "JWT_SECRET_KEY": lambda x: x and len(x) >= 32,
            "XAI_API_KEY": lambda x: x and len(x) >= 16,
            "CORS_ORIGINS": lambda x: True,  # Optional
        }

        for secret_name, validator in essential_secrets.items():
            try:
                value = self.get_secret(secret_name, required=False)
                validation_results[secret_name] = validator(value) if value else False
            except Exception as e:
                validation_results[secret_name] = False
                logger.error(f"Secret validation failed for {secret_name}: {e}")

        return validation_results

    @contextmanager
    def secure_temp_file(self, content: str):
        """Create a secure temporary file for sensitive content"""
        temp_file = None
        try:
            # Create temporary file with restrictive permissions
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tmp')
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())

            # Set restrictive permissions
            os.chmod(temp_file.name, 0o600)

            yield temp_file.name

        finally:
            # Securely delete temporary file
            if temp_file:
                temp_file.close()
                if os.path.exists(temp_file.name):
                    # Overwrite with random data before deletion
                    with open(temp_file.name, 'wb') as f:
                        f.write(os.urandom(len(content)))
                    os.unlink(temp_file.name)

    def audit_environment_variables(self) -> Dict[str, Any]:
        """Audit environment variables for security issues"""
        audit_results = {
            "secrets_in_env": [],
            "insecure_defaults": [],
            "missing_required": [],
            "recommendations": []
        }

        # Check for secrets in environment variables
        for key, value in os.environ.items():
            if any(keyword in key.upper() for keyword in
                   ['SECRET', 'PASSWORD', 'KEY', 'TOKEN', 'CREDENTIAL']):
                if value and len(value) > 10:  # Only flag non-empty secrets
                    audit_results["secrets_in_env"].append(key)

        # Check for insecure defaults
        insecure_defaults = {
            "SECRET_KEY": "changeme",
            "JWT_SECRET": "secret",
            "DATABASE_URL": "sqlite:///./test.db"
        }

        for key, default_val in insecure_defaults.items():
            env_val = os.getenv(key, "")
            if env_val == default_val:
                audit_results["insecure_defaults"].append(key)

        # Recommendations
        if audit_results["secrets_in_env"]:
            audit_results["recommendations"].append(
                "Consider using encrypted secrets storage instead of environment variables"
            )

        if audit_results["insecure_defaults"]:
            audit_results["recommendations"].append(
                "Update default values for security-critical configuration"
            )

        return audit_results


# Global instance
_secrets_manager: Optional[SecretsManager] = None

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

def get_secret(secret_name: str, **kwargs) -> Optional[str]:
    """Convenience function to get a secret"""
    return get_secrets_manager().get_secret(secret_name, **kwargs)

def set_secret(secret_name: str, secret_value: str, **kwargs):
    """Convenience function to set a secret"""
    return get_secrets_manager().set_secret(secret_name, secret_value, **kwargs)


# Export main classes and functions
__all__ = [
    'SecretsManager',
    'SecretConfig',
    'SecretNotFoundError',
    'SecretValidationError',
    'get_secrets_manager',
    'get_secret',
    'set_secret'
]

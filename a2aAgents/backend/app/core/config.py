from typing import List, Union, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, validator
import os
from .secrets import get_secrets_manager, SecretNotFoundError


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Warning: Required environment variables not set for A2A compliance: {missing_vars}")
    # Set defaults for missing variables to allow startup
    for var in missing_vars:
        if var == "A2A_SERVICE_URL":
            os.environ[var] = "http://localhost:4004"
        elif var == "A2A_SERVICE_HOST":
            os.environ[var] = "localhost"
        elif var == "A2A_BASE_URL":
            os.environ[var] = "http://localhost:8000"

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "A2A Platform"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Security - loaded securely from secrets manager
    _secrets_manager = None

    @property
    def SECRET_KEY(self) -> str:
        # First try environment variable
        env_secret = os.getenv("JWT_SECRET_KEY")
        if env_secret:
            return env_secret

        # Then try secrets manager
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_jwt_secret()
        except (SecretNotFoundError, Exception):
            # In production, this should fail, but for initial deployment we'll generate a temporary one
            if os.getenv("A2A_ENVIRONMENT") == "production" and not os.getenv("A2A_ALLOW_TEMP_SECRETS"):
                raise ValueError("JWT_SECRET_KEY must be set in environment variables or secrets storage")
            else:
                # Generate a temporary secret for development/testing
                import secrets
                temp_secret = secrets.token_urlsafe(32)
                print(f"WARNING: Using temporary JWT secret. Set JWT_SECRET_KEY for production use.")
                return temp_secret

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # X.AI Configuration - loaded securely
    @property
    def XAI_API_KEY(self) -> str:
        # First try environment variable
        env_key = os.getenv("XAI_API_KEY")
        if env_key:
            return env_key

        # Then try secrets manager
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_api_key("XAI")
        except (SecretNotFoundError, Exception):
            # For deployment without XAI key, return None and let individual components handle it
            print(f"WARNING: XAI_API_KEY not set. Some AI features will be disabled.")
            return None

    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-4-latest"
    XAI_TIMEOUT: int = 30

    # SQLite Fallback Database
    SQLITE_DB_PATH: str = "./data/a2a_fallback.db"
    SQLITE_JOURNAL_MODE: str = "WAL"

    # A2A Network Configuration
    A2A_NETWORK_PATH: str = os.getenv("A2A_NETWORK_PATH", os.path.join(os.path.dirname(__file__), "../../../../a2aNetwork"))
    A2A_REGISTRY_STORAGE: str = os.getenv("A2A_REGISTRY_STORAGE", "redis")  # Options: redis, etcd, consul, local

    # Redis configuration for distributed storage (with security)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_USE_SSL: bool = os.getenv("REDIS_USE_SSL", "false").lower() == "true"
    REDIS_KEY_PREFIX: str = "a2a:registry:"
    REDIS_TTL: int = 3600  # 1 hour TTL for registry entries

    # Etcd configuration (alternative to Redis)
    ETCD_HOST: str = os.getenv("ETCD_HOST", "localhost")
    ETCD_PORT: int = int(os.getenv("ETCD_PORT", "2379"))

    # Consul configuration (alternative to Redis/Etcd)
    CONSUL_HOST: str = os.getenv("CONSUL_HOST", "localhost")
    CONSUL_PORT: int = int(os.getenv("CONSUL_PORT", "8500"))

    # Service Discovery Configuration
    REGISTRY_URL: str = os.getenv("REGISTRY_URL", "http://localhost:8080")
    TRUST_SERVICE_URL: str = os.getenv("TRUST_SERVICE_URL", "http://localhost:8081")

    # Additional API Keys for various services
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_api_key("OPENAI")
        except SecretNotFoundError:
            return os.getenv("OPENAI_API_KEY")

    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_api_key("ANTHROPIC")
        except SecretNotFoundError:
            return os.getenv("ANTHROPIC_API_KEY")

    # Blockchain Configuration
    BLOCKCHAIN_RPC_URL: str = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
    BLOCKCHAIN_CONTRACT_ADDRESS: Optional[str] = os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS")

    # Network and Service Configuration
    DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))

    # Database - loaded securely
    @property
    def DATABASE_URL(self) -> str:
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_database_url("primary")
        except SecretNotFoundError:
            raise ValueError("DATABASE_URL must be set in environment variables or secrets storage")

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


settings = Settings()

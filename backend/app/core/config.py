from typing import List, Union, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os
from .secrets import get_secrets_manager, SecretNotFoundError


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "FinSight CIB"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security - loaded securely from secrets manager
    _secrets_manager = None
    
    @property
    def SECRET_KEY(self) -> str:
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_jwt_secret()
        except SecretNotFoundError:
            raise ValueError("JWT_SECRET_KEY must be set in environment variables or secrets storage")
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # X.AI Configuration - loaded securely
    @property
    def XAI_API_KEY(self) -> str:
        if not self._secrets_manager:
            self._secrets_manager = get_secrets_manager()
        try:
            return self._secrets_manager.get_api_key("XAI")
        except SecretNotFoundError:
            raise ValueError("XAI_API_KEY must be set in environment variables or secrets storage")
    
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-4-latest"
    XAI_TIMEOUT: int = 30
    
    # SQLite Fallback Database
    SQLITE_DB_PATH: str = "./data/a2a_fallback.db"
    SQLITE_JOURNAL_MODE: str = "WAL"
    
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

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
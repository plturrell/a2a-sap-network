from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "FinSight CIB"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # X.AI Configuration
    XAI_API_KEY: str
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-4-latest"
    XAI_TIMEOUT: int = 30
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # Database
    DATABASE_URL: str
    
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
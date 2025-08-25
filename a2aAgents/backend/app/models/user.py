"""
User model for A2A authentication system
"""

from typing import List, Optional
from pydantic import BaseModel, validator
from datetime import datetime


class User(BaseModel):
    """User model for authentication and authorization"""

    id: str
    username: Optional[str] = None
    email: Optional[str] = None
    tier: str = "authenticated"  # anonymous, authenticated, premium, admin
    scopes: List[str] = []
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None

    @validator('tier')
    def validate_tier(cls, v):
        valid_tiers = {'anonymous', 'authenticated', 'premium', 'admin'}
        if v not in valid_tiers:
            raise ValueError(f'Invalid tier. Must be one of: {valid_tiers}')
        return v

    @validator('scopes')
    def validate_scopes(cls, v):
        # Ensure scopes is always a list
        if isinstance(v, str):
            return [v]
        return v or []

    def has_scope(self, scope: str) -> bool:
        """Check if user has specific scope"""
        return scope in self.scopes

    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return self.tier == "admin" or "admin" in self.scopes

    def is_premium(self) -> bool:
        """Check if user has premium access"""
        return self.tier in ["premium", "admin"]


class UserCreate(BaseModel):
    """Model for user creation"""
    username: str
    email: str
    tier: str = "authenticated"
    scopes: List[str] = []


class UserUpdate(BaseModel):
    """Model for user updates"""
    username: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    scopes: Optional[List[str]] = None


class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str = ""


class APIKey(BaseModel):
    """API Key model"""
    key: str
    name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None

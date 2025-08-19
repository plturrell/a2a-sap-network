"""
Secure Session Management and Token Refresh
Implements secure session handling with JWT refresh tokens
"""

import json
import time
import uuid
import logging
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import secrets
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    
try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import settings
from .errorHandling import AuthenticationError, SecurityError, ValidationError
from .secrets import get_secrets_manager
from .securityMonitoring import report_security_event, EventType, ThreatLevel

logger = logging.getLogger(__name__)

# Password hashing context (optional)
pwd_context = None
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class SessionInfo:
    """Session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.utcnow)
    refresh_token_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    security_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RefreshTokenInfo:
    """Refresh token information"""
    token_id: str
    user_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    used_at: Optional[datetime] = None
    is_revoked: bool = False
    token_family: str = field(default_factory=lambda: str(uuid.uuid4()))
    rotation_count: int = 0


class SessionManager:
    """Secure session management service"""
    
    def __init__(self):
        if not JWT_AVAILABLE:
            logger.warning("JWT library not available - session management will have limited functionality")
            
        self.secrets_manager = get_secrets_manager()
        self.jwt_secret = self.secrets_manager.get_secret(
            "JWT_SECRET_KEY",
            default=settings.SECRET_KEY
        )
        self.jwt_algorithm = "HS256"
        
        # Token expiration settings
        self.access_token_expire_minutes = 15  # Short-lived access tokens
        self.refresh_token_expire_days = 30    # Long-lived refresh tokens
        self.session_expire_hours = 24         # Session lifetime
        
        # Security settings
        self.max_sessions_per_user = 5
        self.require_device_fingerprint = True
        self.enforce_ip_binding = False  # Can be enabled for high security
        self.detect_token_reuse = True   # Detect refresh token replay
        
        # Initialize Redis for session storage
        self._init_redis()
        
        # In-memory caches (for development/fallback)
        self.sessions: Dict[str, SessionInfo] = {}
        self.refresh_tokens: Dict[str, RefreshTokenInfo] = {}
        self.revoked_tokens: Set[str] = set()
        
        logger.info("Session Manager initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not available, using in-memory storage")
            self.redis_client = None
            self.use_redis = False
            return
            
        try:
            redis_url = self.secrets_manager.get_secret(
                "REDIS_URL",
                default="redis://localhost:6379/1"
            )
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis connected for session storage")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None
            self.use_redis = False
    
    async def create_session(self,
                           user_id: str,
                           ip_address: str,
                           user_agent: str,
                           device_fingerprint: Optional[str] = None) -> Tuple[str, str, SessionInfo]:
        """
        Create a new session with tokens
        
        Returns: (access_token, refresh_token, session_info)
        """
        try:
            # Check maximum sessions
            active_sessions = await self.get_user_sessions(user_id)
            if len(active_sessions) >= self.max_sessions_per_user:
                # Terminate oldest session
                oldest = min(active_sessions, key=lambda s: s.created_at)
                await self.terminate_session(oldest.session_id)
                
                await report_security_event(
                    EventType.ACCESS_DENIED,
                    ThreatLevel.LOW,
                    f"Maximum sessions exceeded for user {user_id}, terminated oldest",
                    user_id=user_id,
                    source_ip=ip_address
                )
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create session
            session = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=self.session_expire_hours),
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                security_flags={
                    "mfa_verified": False,
                    "password_change_required": False,
                    "suspicious_activity": False
                }
            )
            
            # Generate tokens
            access_token = self._create_access_token(user_id, session_id)
            refresh_token, refresh_info = await self._create_refresh_token(user_id, session_id)
            
            # Store session
            session.refresh_token_id = refresh_info.token_id
            await self._store_session(session)
            
            # Log session creation
            await report_security_event(
                EventType.LOGIN_SUCCESS,
                ThreatLevel.INFO,
                f"New session created for user {user_id}",
                user_id=user_id,
                source_ip=ip_address,
                session_id=session_id
            )
            
            return access_token, refresh_token, session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SecurityError("Session creation failed")
    
    def _create_access_token(self, user_id: str, session_id: str) -> str:
        """Create a short-lived access token"""
        payload = {
            "sub": user_id,
            "sid": session_id,
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
            "jti": str(uuid.uuid4())  # Unique token ID
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _create_refresh_token(self, user_id: str, session_id: str) -> Tuple[str, RefreshTokenInfo]:
        """Create a long-lived refresh token"""
        token_id = str(uuid.uuid4())
        
        # Create token info
        refresh_info = RefreshTokenInfo(
            token_id=token_id,
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        )
        
        # Create JWT
        payload = {
            "sub": user_id,
            "sid": session_id,
            "tid": token_id,
            "type": "refresh",
            "fam": refresh_info.token_family,
            "iat": refresh_info.created_at,
            "exp": refresh_info.expires_at
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store refresh token info
        await self._store_refresh_token(refresh_info)
        
        return token, refresh_info
    
    async def refresh_access_token(self,
                                 refresh_token: str,
                                 ip_address: str,
                                 user_agent: str) -> Tuple[str, str]:
        """
        Refresh access token using refresh token
        Implements token rotation for security
        
        Returns: (new_access_token, new_refresh_token)
        """
        try:
            # Decode refresh token
            try:
                payload = jwt.decode(
                    refresh_token,
                    self.jwt_secret,
                    algorithms=[self.jwt_algorithm]
                )
            except jwt.ExpiredSignatureError:
                raise AuthenticationError("Refresh token has expired")
            except jwt.InvalidTokenError:
                raise AuthenticationError("Invalid refresh token")
            
            # Validate token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")
            
            token_id = payload.get("tid")
            session_id = payload.get("sid")
            user_id = payload.get("sub")
            token_family = payload.get("fam")
            
            # Get refresh token info
            refresh_info = await self._get_refresh_token(token_id)
            if not refresh_info:
                raise AuthenticationError("Refresh token not found")
            
            # Check if token is revoked
            if refresh_info.is_revoked:
                # Potential token theft - revoke entire family
                await self._revoke_token_family(token_family)
                
                await report_security_event(
                    EventType.SUSPICIOUS_TRAFFIC,
                    ThreatLevel.HIGH,
                    f"Attempted use of revoked refresh token - possible token theft",
                    user_id=user_id,
                    source_ip=ip_address,
                    session_id=session_id
                )
                
                raise AuthenticationError("Token has been revoked")
            
            # Check if token has been used (replay detection)
            if self.detect_token_reuse and refresh_info.used_at:
                # Token reuse detected - revoke entire family
                await self._revoke_token_family(token_family)
                
                await report_security_event(
                    EventType.SUSPICIOUS_TRAFFIC,
                    ThreatLevel.CRITICAL,
                    f"Refresh token reuse detected - likely token theft",
                    user_id=user_id,
                    source_ip=ip_address,
                    session_id=session_id
                )
                
                raise AuthenticationError("Token reuse detected - security breach")
            
            # Get session
            session = await self._get_session(session_id)
            if not session or not session.is_active:
                raise AuthenticationError("Session not found or inactive")
            
            # Validate session security
            if self.enforce_ip_binding and session.ip_address != ip_address:
                await report_security_event(
                    EventType.SUSPICIOUS_TRAFFIC,
                    ThreatLevel.HIGH,
                    f"Session IP mismatch: expected {session.ip_address}, got {ip_address}",
                    user_id=user_id,
                    source_ip=ip_address,
                    session_id=session_id
                )
                raise AuthenticationError("Session IP mismatch")
            
            # Mark old refresh token as used
            refresh_info.used_at = datetime.utcnow()
            await self._store_refresh_token(refresh_info)
            
            # Rotate refresh token (create new one, invalidate old)
            new_refresh_token, new_refresh_info = await self._create_refresh_token(
                user_id, session_id
            )
            new_refresh_info.token_family = token_family
            new_refresh_info.rotation_count = refresh_info.rotation_count + 1
            await self._store_refresh_token(new_refresh_info)
            
            # Create new access token
            new_access_token = self._create_access_token(user_id, session_id)
            
            # Update session
            session.last_activity = datetime.utcnow()
            session.refresh_token_id = new_refresh_info.token_id
            await self._store_session(session)
            
            logger.info(f"Tokens refreshed for user {user_id}, session {session_id}")
            
            return new_access_token, new_refresh_token
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError("Token refresh failed")
    
    async def validate_access_token(self, token: str) -> Dict[str, Any]:
        """Validate access token and return claims"""
        try:
            # Check if token is revoked
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if await self._is_token_revoked(token_hash):
                raise AuthenticationError("Token has been revoked")
            
            # Decode token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Validate token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            # Get session
            session_id = payload.get("sid")
            session = await self._get_session(session_id)
            
            if not session or not session.is_active:
                raise AuthenticationError("Session not found or inactive")
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            await self._store_session(session)
            
            return {
                "user_id": payload.get("sub"),
                "session_id": session_id,
                "session": session
            }
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Access token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid access token")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise AuthenticationError("Token validation failed")
    
    async def terminate_session(self, session_id: str, reason: str = "User logout"):
        """Terminate a session and revoke associated tokens"""
        try:
            session = await self._get_session(session_id)
            if session:
                # Mark session as inactive
                session.is_active = False
                await self._store_session(session)
                
                # Revoke refresh token
                if session.refresh_token_id:
                    refresh_info = await self._get_refresh_token(session.refresh_token_id)
                    if refresh_info:
                        refresh_info.is_revoked = True
                        await self._store_refresh_token(refresh_info)
                
                # Log termination
                await report_security_event(
                    EventType.LOGIN_SUCCESS,  # Using as logout event
                    ThreatLevel.INFO,
                    f"Session terminated: {reason}",
                    user_id=session.user_id,
                    session_id=session_id
                )
                
                logger.info(f"Session {session_id} terminated: {reason}")
                
        except Exception as e:
            logger.error(f"Failed to terminate session: {e}")
    
    async def terminate_all_user_sessions(self, user_id: str, reason: str = "Security measure"):
        """Terminate all sessions for a user"""
        sessions = await self.get_user_sessions(user_id)
        for session in sessions:
            await self.terminate_session(session.session_id, reason)
        
        await report_security_event(
            EventType.ACCESS_DENIED,
            ThreatLevel.MEDIUM,
            f"All sessions terminated for user {user_id}: {reason}",
            user_id=user_id
        )
    
    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user"""
        if self.use_redis:
            # Get from Redis
            session_keys = self.redis_client.keys(f"session:*:user:{user_id}")
            sessions = []
            for key in session_keys:
                session_data = self.redis_client.get(key)
                if session_data:
                    session = self._deserialize_session(json.loads(session_data))
                    if session.is_active and session.expires_at > datetime.utcnow():
                        sessions.append(session)
            return sessions
        else:
            # Get from memory
            return [
                session for session in self.sessions.values()
                if session.user_id == user_id and 
                   session.is_active and 
                   session.expires_at > datetime.utcnow()
            ]
    
    async def revoke_token(self, token: str):
        """Revoke a specific token"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        if self.use_redis:
            self.redis_client.setex(
                f"revoked_token:{token_hash}",
                timedelta(days=self.refresh_token_expire_days),
                "1"
            )
        else:
            self.revoked_tokens.add(token_hash)
    
    # Storage methods
    async def _store_session(self, session: SessionInfo):
        """Store session information"""
        if self.use_redis:
            key = f"session:{session.session_id}:user:{session.user_id}"
            ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(self._serialize_session(session))
            )
        else:
            self.sessions[session.session_id] = session
    
    async def _get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        if self.use_redis:
            # Try to find session key
            keys = self.redis_client.keys(f"session:{session_id}:user:*")
            if keys:
                session_data = self.redis_client.get(keys[0])
                if session_data:
                    return self._deserialize_session(json.loads(session_data))
            return None
        else:
            return self.sessions.get(session_id)
    
    async def _store_refresh_token(self, refresh_info: RefreshTokenInfo):
        """Store refresh token information"""
        if self.use_redis:
            key = f"refresh_token:{refresh_info.token_id}"
            ttl = int((refresh_info.expires_at - datetime.utcnow()).total_seconds())
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(self._serialize_refresh_token(refresh_info))
            )
        else:
            self.refresh_tokens[refresh_info.token_id] = refresh_info
    
    async def _get_refresh_token(self, token_id: str) -> Optional[RefreshTokenInfo]:
        """Get refresh token information"""
        if self.use_redis:
            token_data = self.redis_client.get(f"refresh_token:{token_id}")
            if token_data:
                return self._deserialize_refresh_token(json.loads(token_data))
            return None
        else:
            return self.refresh_tokens.get(token_id)
    
    async def _is_token_revoked(self, token_hash: str) -> bool:
        """Check if token is revoked"""
        if self.use_redis:
            return bool(self.redis_client.get(f"revoked_token:{token_hash}"))
        else:
            return token_hash in self.revoked_tokens
    
    async def _revoke_token_family(self, token_family: str):
        """Revoke entire token family (for security breach scenarios)"""
        logger.warning(f"Revoking entire token family: {token_family}")
        
        if self.use_redis:
            # Find all tokens in family
            for key in self.redis_client.scan_iter("refresh_token:*"):
                token_data = self.redis_client.get(key)
                if token_data:
                    refresh_info = self._deserialize_refresh_token(json.loads(token_data))
                    if refresh_info.token_family == token_family:
                        refresh_info.is_revoked = True
                        await self._store_refresh_token(refresh_info)
        else:
            # Revoke in memory
            for refresh_info in self.refresh_tokens.values():
                if refresh_info.token_family == token_family:
                    refresh_info.is_revoked = True
    
    # Serialization helpers
    async def create_password_reset_token(self, user_id: str, email: str) -> str:
        """Create secure password reset token valid for 1 hour"""
        try:
            # Generate secure random token
            reset_token = secrets.token_urlsafe(32)
            
            # Create token data
            token_data = {
                "user_id": user_id,
                "email": email,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "token_type": "password_reset"
            }
            
            # Store token (in production, use Redis with TTL)
            if not hasattr(self, '_password_reset_tokens'):
                self._password_reset_tokens = {}
            
            self._password_reset_tokens[reset_token] = token_data
            
            logger.info(f"Password reset token created for user {user_id}")
            return reset_token
            
        except Exception as e:
            logger.error(f"Failed to create password reset token: {e}")
            raise SecurityError("Failed to create password reset token")
    
    async def validate_password_reset_token(self, reset_token: str) -> Dict[str, Any]:
        """Validate password reset token and return user data"""
        try:
            if not hasattr(self, '_password_reset_tokens'):
                self._password_reset_tokens = {}
            
            token_data = self._password_reset_tokens.get(reset_token)
            
            if not token_data:
                raise ValidationError("Invalid reset token")
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.utcnow() > expires_at:
                # Clean up expired token
                del self._password_reset_tokens[reset_token]
                raise ValidationError("Reset token has expired")
            
            return token_data
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to validate password reset token: {e}")
            raise ValidationError("Invalid reset token")
    
    async def invalidate_password_reset_token(self, reset_token: str):
        """Invalidate used password reset token"""
        try:
            if hasattr(self, '_password_reset_tokens') and reset_token in self._password_reset_tokens:
                del self._password_reset_tokens[reset_token]
                logger.info("Password reset token invalidated")
        except Exception as e:
            logger.error(f"Failed to invalidate password reset token: {e}")

    def _serialize_session(self, session: SessionInfo) -> Dict[str, Any]:
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "is_active": session.is_active,
            "last_activity": session.last_activity.isoformat(),
            "refresh_token_id": session.refresh_token_id,
            "device_fingerprint": session.device_fingerprint,
            "security_flags": session.security_flags
        }
    
    def _deserialize_session(self, data: Dict[str, Any]) -> SessionInfo:
        return SessionInfo(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            is_active=data["is_active"],
            last_activity=datetime.fromisoformat(data["last_activity"]),
            refresh_token_id=data.get("refresh_token_id"),
            device_fingerprint=data.get("device_fingerprint"),
            security_flags=data.get("security_flags", {})
        )
    
    def _serialize_refresh_token(self, refresh_info: RefreshTokenInfo) -> Dict[str, Any]:
        return {
            "token_id": refresh_info.token_id,
            "user_id": refresh_info.user_id,
            "session_id": refresh_info.session_id,
            "created_at": refresh_info.created_at.isoformat(),
            "expires_at": refresh_info.expires_at.isoformat(),
            "used_at": refresh_info.used_at.isoformat() if refresh_info.used_at else None,
            "is_revoked": refresh_info.is_revoked,
            "token_family": refresh_info.token_family,
            "rotation_count": refresh_info.rotation_count
        }
    
    def _deserialize_refresh_token(self, data: Dict[str, Any]) -> RefreshTokenInfo:
        return RefreshTokenInfo(
            token_id=data["token_id"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            used_at=datetime.fromisoformat(data["used_at"]) if data.get("used_at") else None,
            is_revoked=data["is_revoked"],
            token_family=data["token_family"],
            rotation_count=data.get("rotation_count", 0)
        )


# Global instance
_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Export main classes and functions
__all__ = [
    'SessionManager',
    'SessionInfo',
    'RefreshTokenInfo',
    'get_session_manager'
]
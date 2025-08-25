"""
Session Management Service for SAP BTP Integration
Provides secure session handling with XSUAA token management
"""

import asyncio
import json
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from uuid import uuid4
import logging
import redis
import pickle

from pydantic import BaseModel, Field
from fastapi import HTTPException, Request, Response
from .rbacService import UserInfo

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    INVALID = "invalid"


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    user_info: UserInfo
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE

    # Session metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    login_method: str = "xsuaa"

    # Token information
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None

    # Activity tracking
    last_activity: Dict[str, Any] = Field(default_factory=dict)
    activity_count: int = 0


class SessionActivity(BaseModel):
    """Session activity record"""
    session_id: str
    activity_type: str  # login, logout, api_call, page_view, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SessionService:
    """Session management service for SAP BTP applications"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_timeout = timedelta(minutes=config.get("session_timeout_minutes", 30))
        self.max_sessions_per_user = config.get("max_sessions_per_user", 5)
        self.cleanup_interval = timedelta(minutes=config.get("cleanup_interval_minutes", 5))

        # Initialize Redis for session storage
        redis_config = config.get("redis", {})
        self.redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            decode_responses=False,  # We'll handle encoding ourselves
            socket_timeout=redis_config.get("socket_timeout", 5),
            socket_connect_timeout=redis_config.get("socket_connect_timeout", 5),
            retry_on_timeout=True,
            max_connections=redis_config.get("max_connections", 10)
        )

        # In-memory fallback if Redis is not available
        self.memory_sessions: Dict[str, SessionInfo] = {}
        self.use_redis = True

        try:
            self.redis_client.ping()
            logger.info("Session Service initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory session storage: {e}")
            self.use_redis = False

        # Start cleanup task (if enabled)
        if config.get("enable_cleanup_task", True):
            asyncio.create_task(self._cleanup_expired_sessions())

    async def create_session(
        self,
        user_info: UserInfo,
        request: Request,
        access_token: str,
        refresh_token: Optional[str] = None
    ) -> SessionInfo:
        """Create new user session"""
        try:
            # Generate session ID
            session_id = str(uuid4())

            # Extract request metadata
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent", "")

            # Calculate expiration
            expires_at = datetime.utcnow() + self.session_timeout

            # Decode token to get expiration
            token_expires_at = None
            try:
                decoded_token = jwt.decode(access_token, options={"verify_signature": False})
                if "exp" in decoded_token:
                    token_expires_at = datetime.fromtimestamp(decoded_token["exp"])
            except Exception:
                pass

            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                user_info=user_info,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expires_at=token_expires_at
            )

            # Check session limits
            await self._enforce_session_limits(user_info.user_id)

            # Store session
            await self._store_session(session_info)

            # Log session creation
            await self._log_activity(SessionActivity(
                session_id=session_id,
                activity_type="login",
                details={
                    "user_id": user_info.user_id,
                    "user_name": user_info.user_name,
                    "login_method": "xsuaa"
                },
                ip_address=ip_address,
                user_agent=user_agent
            ))

            logger.info(f"Created session for user {user_info.user_name}: {session_id}")
            return session_info

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise HTTPException(status_code=500, detail="Failed to create session")

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        try:
            session_info = await self._retrieve_session(session_id)

            if not session_info:
                return None

            # Check if session is terminated
            if session_info.status == SessionStatus.TERMINATED:
                return None

            # Check if session is expired
            if datetime.utcnow() > session_info.expires_at:
                session_info.status = SessionStatus.EXPIRED
                await self._store_session(session_info)
                return None

            # Update last accessed time
            session_info.last_accessed = datetime.utcnow()
            session_info.activity_count += 1
            await self._store_session(session_info)

            return session_info

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def refresh_session(self, session_id: str) -> Optional[SessionInfo]:
        """Refresh session expiration"""
        try:
            session_info = await self.get_session(session_id)

            if not session_info or session_info.status != SessionStatus.ACTIVE:
                return None

            # Extend session expiration
            session_info.expires_at = datetime.utcnow() + self.session_timeout
            session_info.last_accessed = datetime.utcnow()

            await self._store_session(session_info)

            logger.debug(f"Refreshed session: {session_id}")
            return session_info

        except Exception as e:
            logger.error(f"Failed to refresh session {session_id}: {e}")
            return None

    async def terminate_session(self, session_id: str) -> bool:
        """Terminate session"""
        try:
            session_info = await self._retrieve_session(session_id)

            if not session_info:
                return False

            # Mark session as terminated
            session_info.status = SessionStatus.TERMINATED
            session_info.last_accessed = datetime.utcnow()

            await self._store_session(session_info)

            # Log session termination
            await self._log_activity(SessionActivity(
                session_id=session_id,
                activity_type="logout",
                details={
                    "user_id": session_info.user_info.user_id,
                    "termination_reason": "user_logout"
                }
            ))

            logger.info(f"Terminated session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False

    async def terminate_user_sessions(self, user_id: str, exclude_session: Optional[str] = None) -> int:
        """Terminate all sessions for a user"""
        try:
            user_sessions = await self.get_user_sessions(user_id)
            terminated_count = 0

            for session_info in user_sessions:
                if exclude_session and session_info.session_id == exclude_session:
                    continue

                if await self.terminate_session(session_info.session_id):
                    terminated_count += 1

            logger.info(f"Terminated {terminated_count} sessions for user {user_id}")
            return terminated_count

        except Exception as e:
            logger.error(f"Failed to terminate user sessions for {user_id}: {e}")
            return 0

    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user"""
        try:
            all_sessions = await self._get_all_sessions()
            user_sessions = []

            for session_info in all_sessions:
                if (session_info.user_info.user_id == user_id and
                    session_info.status == SessionStatus.ACTIVE and
                    datetime.utcnow() <= session_info.expires_at):
                    user_sessions.append(session_info)

            return user_sessions

        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []

    async def update_session_activity(
        self,
        session_id: str,
        activity_type: str,
        details: Dict[str, Any] = None
    ):
        """Update session activity"""
        try:
            session_info = await self.get_session(session_id)

            if session_info:
                session_info.last_activity = {
                    "type": activity_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": details or {}
                }

                await self._store_session(session_info)

                # Log activity
                await self._log_activity(SessionActivity(
                    session_id=session_id,
                    activity_type=activity_type,
                    details=details or {}
                ))

        except Exception as e:
            logger.error(f"Failed to update session activity for {session_id}: {e}")

    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            all_sessions = await self._get_all_sessions()

            active_sessions = [s for s in all_sessions if s.status == SessionStatus.ACTIVE]
            expired_sessions = [s for s in all_sessions if s.status == SessionStatus.EXPIRED]
            terminated_sessions = [s for s in all_sessions if s.status == SessionStatus.TERMINATED]

            # Calculate statistics
            stats = {
                "total_sessions": len(all_sessions),
                "active_sessions": len(active_sessions),
                "expired_sessions": len(expired_sessions),
                "terminated_sessions": len(terminated_sessions),
                "unique_users": len(set(s.user_info.user_id for s in active_sessions)),
                "average_session_duration": self._calculate_average_session_duration(all_sessions),
                "most_active_users": self._get_most_active_users(active_sessions),
                "session_by_hour": self._get_sessions_by_hour(all_sessions)
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}

    async def _store_session(self, session_info: SessionInfo):
        """Store session information"""
        try:
            if self.use_redis:
                # Store in Redis
                session_data = pickle.dumps(session_info.dict())
                key = f"session:{session_info.session_id}"

                # Set with expiration
                ttl_seconds = int((session_info.expires_at - datetime.utcnow()).total_seconds())
                if ttl_seconds > 0:
                    self.redis_client.setex(key, ttl_seconds, session_data)

                # Store user session mapping
                user_key = f"user_sessions:{session_info.user_info.user_id}"
                self.redis_client.sadd(user_key, session_info.session_id)
                self.redis_client.expire(user_key, ttl_seconds)
            else:
                # Store in memory
                self.memory_sessions[session_info.session_id] = session_info

        except Exception as e:
            logger.error(f"Failed to store session {session_info.session_id}: {e}")
            raise

    async def _retrieve_session(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieve session information"""
        try:
            if self.use_redis:
                # Retrieve from Redis
                key = f"session:{session_id}"
                session_data = self.redis_client.get(key)

                if session_data:
                    session_dict = pickle.loads(session_data)
                    return SessionInfo(**session_dict)
            else:
                # Retrieve from memory
                return self.memory_sessions.get(session_id)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve session {session_id}: {e}")
            return None

    async def _get_all_sessions(self) -> List[SessionInfo]:
        """Get all sessions"""
        try:
            sessions = []

            if self.use_redis:
                # Get all session keys
                keys = self.redis_client.keys("session:*")

                for key in keys:
                    session_data = self.redis_client.get(key)
                    if session_data:
                        session_dict = pickle.loads(session_data)
                        sessions.append(SessionInfo(**session_dict))
            else:
                # Get from memory
                sessions = list(self.memory_sessions.values())

            return sessions

        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []

    async def _enforce_session_limits(self, user_id: str):
        """Enforce session limits per user"""
        try:
            user_sessions = await self.get_user_sessions(user_id)

            if len(user_sessions) >= self.max_sessions_per_user:
                # Terminate oldest sessions
                user_sessions.sort(key=lambda x: x.created_at)
                sessions_to_terminate = user_sessions[:-self.max_sessions_per_user + 1]

                for session in sessions_to_terminate:
                    await self.terminate_session(session.session_id)

        except Exception as e:
            logger.error(f"Failed to enforce session limits for user {user_id}: {e}")

    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())

                all_sessions = await self._get_all_sessions()
                expired_count = 0

                for session in all_sessions:
                    if (datetime.utcnow() > session.expires_at and
                        session.status == SessionStatus.ACTIVE):
                        session.status = SessionStatus.EXPIRED
                        await self._store_session(session)
                        expired_count += 1

                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired sessions")

            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")

    async def _log_activity(self, activity: SessionActivity):
        """Log session activity"""
        try:
            # In a production system, this would write to a proper audit log
            logger.info(f"Session activity: {activity.activity_type} for session {activity.session_id}")

        except Exception as e:
            logger.error(f"Failed to log session activity: {e}")

    def _calculate_average_session_duration(self, sessions: List[SessionInfo]) -> float:
        """Calculate average session duration in minutes"""
        if not sessions:
            return 0.0

        total_duration = 0.0
        count = 0

        for session in sessions:
            if session.status in [SessionStatus.EXPIRED, SessionStatus.TERMINATED]:
                duration = (session.last_accessed - session.created_at).total_seconds() / 60
                total_duration += duration
                count += 1

        return total_duration / count if count > 0 else 0.0

    def _get_most_active_users(self, sessions: List[SessionInfo], limit: int = 5) -> List[Dict[str, Any]]:
        """Get most active users by session count"""
        user_activity = {}

        for session in sessions:
            user_id = session.user_info.user_id
            if user_id not in user_activity:
                user_activity[user_id] = {
                    "user_id": user_id,
                    "user_name": session.user_info.user_name,
                    "session_count": 0,
                    "total_activity": 0
                }

            user_activity[user_id]["session_count"] += 1
            user_activity[user_id]["total_activity"] += session.activity_count

        # Sort by activity and return top users
        sorted_users = sorted(
            user_activity.values(),
            key=lambda x: x["total_activity"],
            reverse=True
        )

        return sorted_users[:limit]

    def _get_sessions_by_hour(self, sessions: List[SessionInfo]) -> Dict[str, int]:
        """Get session distribution by hour of day"""
        hourly_distribution = {str(i).zfill(2): 0 for i in range(24)}

        for session in sessions:
            hour = session.created_at.strftime("%H")
            hourly_distribution[hour] += 1

        return hourly_distribution


# Global session service instance
session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get session service instance"""
    if session_service is None:
        raise HTTPException(status_code=500, detail="Session service not initialized")
    return session_service


def initialize_session_service(config: Dict[str, Any]):
    """Initialize the global session service"""
    global session_service
    session_service = SessionService(config)
    logger.info("Session Service initialized globally")

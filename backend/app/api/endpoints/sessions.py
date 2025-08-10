"""
Session Management API Endpoints
Manage user sessions and token refresh
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import logging

from ...core.sessionManagement import get_session_manager
from ...core.securityMonitoring import report_security_event, EventType, ThreatLevel
from ..deps import get_current_user
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


class TokenRefreshRequest(BaseModel):
    """Request model for token refresh"""
    refresh_token: str = Field(..., description="The refresh token")


class TokenResponse(BaseModel):
    """Response model for token operations"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token expiration in seconds")


class SessionResponse(BaseModel):
    """Response model for session information"""
    session_id: str
    created_at: str
    expires_at: str
    last_activity: str
    ip_address: str
    user_agent: str
    is_active: bool
    device_fingerprint: Optional[str] = None
    security_flags: Dict[str, bool]


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    token_request: TokenRefreshRequest
) -> TokenResponse:
    """
    Refresh access token using refresh token
    
    - Implements secure token rotation
    - Old refresh token is invalidated
    - Returns new access and refresh tokens
    """
    try:
        session_manager = get_session_manager()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Refresh tokens
        access_token, refresh_token = await session_manager.refresh_access_token(
            token_request.refresh_token,
            client_ip,
            user_agent
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=session_manager.access_token_expire_minutes * 60
        )
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        # Don't reveal specific error details
        raise HTTPException(status_code=401, detail="Token refresh failed")


@router.get("/sessions", response_model=List[SessionResponse])
async def get_my_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[SessionResponse]:
    """
    Get all active sessions for the current user
    
    Returns list of active sessions with details.
    """
    try:
        session_manager = get_session_manager()
        sessions = await session_manager.get_user_sessions(current_user["user_id"])
        
        return [
            SessionResponse(
                session_id=session.session_id,
                created_at=session.created_at.isoformat(),
                expires_at=session.expires_at.isoformat(),
                last_activity=session.last_activity.isoformat(),
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                is_active=session.is_active,
                device_fingerprint=session.device_fingerprint,
                security_flags=session.security_flags
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@router.get("/sessions/current", response_model=SessionResponse)
async def get_current_session(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> SessionResponse:
    """
    Get information about the current session
    """
    try:
        session_manager = get_session_manager()
        
        # Validate token and get session info
        token_info = await session_manager.validate_access_token(credentials.credentials)
        session = token_info["session"]
        
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            expires_at=session.expires_at.isoformat(),
            last_activity=session.last_activity.isoformat(),
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            is_active=session.is_active,
            device_fingerprint=session.device_fingerprint,
            security_flags=session.security_flags
        )
        
    except Exception as e:
        logger.error(f"Failed to get current session: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/sessions/{session_id}/terminate")
async def terminate_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Terminate a specific session
    
    - Invalidates the session
    - Revokes associated tokens
    - Can only terminate own sessions
    """
    try:
        session_manager = get_session_manager()
        
        # Verify session belongs to user
        sessions = await session_manager.get_user_sessions(current_user["user_id"])
        session_ids = [s.session_id for s in sessions]
        
        if session_id not in session_ids:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Terminate session
        await session_manager.terminate_session(
            session_id, 
            reason=f"User requested termination"
        )
        
        return {
            "status": "success",
            "message": "Session terminated successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate session: {e}")
        raise HTTPException(status_code=500, detail="Failed to terminate session")


@router.post("/sessions/terminate-all")
async def terminate_all_sessions(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Terminate all sessions for the current user
    
    - Useful for security incidents
    - Forces re-authentication on all devices
    """
    try:
        session_manager = get_session_manager()
        
        # Get current session ID to exclude it
        auth_header = request.headers.get("authorization", "")
        current_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
        
        current_session_id = None
        if current_token:
            try:
                token_info = await session_manager.validate_access_token(current_token)
                current_session_id = token_info["session_id"]
            except:
                pass
        
        # Get all sessions
        sessions = await session_manager.get_user_sessions(current_user["user_id"])
        terminated_count = 0
        
        # Terminate all except current
        for session in sessions:
            if session.session_id != current_session_id:
                await session_manager.terminate_session(
                    session.session_id,
                    reason="User requested termination of all sessions"
                )
                terminated_count += 1
        
        # Log security event
        await report_security_event(
            EventType.ACCESS_DENIED,
            ThreatLevel.MEDIUM,
            f"User terminated all sessions except current",
            user_id=current_user["user_id"],
            details={"terminated_count": terminated_count}
        )
        
        return {
            "status": "success",
            "message": f"Terminated {terminated_count} sessions",
            "terminated_count": terminated_count
        }
        
    except Exception as e:
        logger.error(f"Failed to terminate all sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to terminate sessions")


@router.post("/sessions/security-check")
async def security_check(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Perform security check on current session
    
    - Validates session security flags
    - Checks for suspicious activity
    - Returns security recommendations
    """
    try:
        session_manager = get_session_manager()
        
        # Get current session
        auth_header = request.headers.get("authorization", "")
        current_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
        
        if not current_token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        token_info = await session_manager.validate_access_token(current_token)
        session = token_info["session"]
        
        # Perform security checks
        security_issues = []
        recommendations = []
        
        # Check MFA
        if not session.security_flags.get("mfa_verified", False):
            security_issues.append("Multi-factor authentication not enabled")
            recommendations.append("Enable MFA for enhanced security")
        
        # Check password change requirement
        if session.security_flags.get("password_change_required", False):
            security_issues.append("Password change required")
            recommendations.append("Change your password immediately")
        
        # Check suspicious activity flag
        if session.security_flags.get("suspicious_activity", False):
            security_issues.append("Suspicious activity detected on account")
            recommendations.append("Review recent account activity")
            recommendations.append("Consider changing your password")
        
        # Check session age
        session_age = (datetime.utcnow() - session.created_at).total_seconds() / 3600
        if session_age > 24:
            recommendations.append("Session is over 24 hours old - consider re-authenticating")
        
        # Check for multiple active sessions
        all_sessions = await session_manager.get_user_sessions(current_user.id)
        if len(all_sessions) > 3:
            recommendations.append(f"You have {len(all_sessions)} active sessions - review and terminate unused ones")
        
        return {
            "status": "checked",
            "session_secure": len(security_issues) == 0,
            "security_issues": security_issues,
            "recommendations": recommendations,
            "session_info": {
                "age_hours": round(session_age, 1),
                "active_sessions": len(all_sessions),
                "mfa_enabled": session.security_flags.get("mfa_verified", False)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security check failed: {e}")
        raise HTTPException(status_code=500, detail="Security check failed")


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Logout current session
    
    - Terminates current session
    - Invalidates tokens
    - Clears any session cookies
    """
    try:
        session_manager = get_session_manager()
        
        # Get current session
        auth_header = request.headers.get("authorization", "")
        current_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
        
        if current_token:
            try:
                token_info = await session_manager.validate_access_token(current_token)
                session_id = token_info["session_id"]
                
                # Terminate session
                await session_manager.terminate_session(session_id, reason="User logout")
                
                # Revoke the access token
                await session_manager.revoke_token(current_token)
                
            except Exception as e:
                logger.warning(f"Error during logout: {e}")
        
        # Clear any session cookies
        response.delete_cookie("session")
        response.delete_cookie("refresh_token")
        
        return {
            "status": "success",
            "message": "Logged out successfully"
        }
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        # Even if logout fails, return success to client
        return {
            "status": "success",
            "message": "Logged out"
        }


# Export router
__all__ = ["router"]
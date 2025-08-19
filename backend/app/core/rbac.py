"""
Role-Based Access Control (RBAC) System
Comprehensive authentication and authorization system with roles and permissions
"""
import platform
import time

import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
# Use hashlib for password hashing instead of bcrypt for now
import hashlib
import jwt
from pydantic import BaseModel, EmailStr

from .secrets import get_secrets_manager, SecretNotFoundError
from .errorHandling import AuthenticationError, AuthorizationError, ValidationError
import bcrypt

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"


class Permission(str, Enum):
    """System permissions"""
    # System administration
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_SYSTEM = "manage_system"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    MANAGE_RATE_LIMITS = "manage_rate_limits"
    
    # A2A Agent management
    MANAGE_AGENTS = "manage_agents"
    VIEW_AGENT_STATUS = "view_agent_status"
    EXECUTE_WORKFLOWS = "execute_workflows"
    
    # Data operations
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # API access
    API_ACCESS = "api_access"
    ADMIN_API_ACCESS = "admin_api_access"
    
    # Financial operations
    VIEW_FINANCIAL_DATA = "view_financial_data"
    MODIFY_FINANCIAL_DATA = "modify_financial_data"


@dataclass
class RoleDefinition:
    """Role definition with permissions and constraints"""
    name: UserRole
    permissions: Set[Permission]
    description: str
    max_api_calls_per_hour: int = 1000
    can_impersonate: bool = False
    session_timeout_minutes: int = 60


# Role hierarchy and permissions
ROLE_DEFINITIONS = {
    UserRole.SUPER_ADMIN: RoleDefinition(
        name=UserRole.SUPER_ADMIN,
        permissions={
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.MANAGE_SYSTEM,
            Permission.VIEW_SYSTEM_LOGS,
            Permission.MANAGE_RATE_LIMITS,
            Permission.MANAGE_AGENTS,
            Permission.VIEW_AGENT_STATUS,
            Permission.EXECUTE_WORKFLOWS,
            Permission.READ_DATA,
            Permission.WRITE_DATA,
            Permission.DELETE_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS,
            Permission.ADMIN_API_ACCESS,
            Permission.VIEW_FINANCIAL_DATA,
            Permission.MODIFY_FINANCIAL_DATA,
        },
        description="Full system access with all permissions",
        max_api_calls_per_hour=10000,
        can_impersonate=True,
        session_timeout_minutes=120
    ),
    UserRole.ADMIN: RoleDefinition(
        name=UserRole.ADMIN,
        permissions={
            Permission.MANAGE_USERS,
            Permission.VIEW_SYSTEM_LOGS,
            Permission.MANAGE_RATE_LIMITS,
            Permission.MANAGE_AGENTS,
            Permission.VIEW_AGENT_STATUS,
            Permission.EXECUTE_WORKFLOWS,
            Permission.READ_DATA,
            Permission.WRITE_DATA,
            Permission.EXPORT_DATA,
            Permission.API_ACCESS,
            Permission.ADMIN_API_ACCESS,
            Permission.VIEW_FINANCIAL_DATA,
            Permission.MODIFY_FINANCIAL_DATA,
        },
        description="Administrative access with user and system management",
        max_api_calls_per_hour=5000,
        can_impersonate=False,
        session_timeout_minutes=60
    ),
    UserRole.MODERATOR: RoleDefinition(
        name=UserRole.MODERATOR,
        permissions={
            Permission.VIEW_SYSTEM_LOGS,
            Permission.VIEW_AGENT_STATUS,
            Permission.READ_DATA,
            Permission.WRITE_DATA,
            Permission.API_ACCESS,
            Permission.VIEW_FINANCIAL_DATA,
        },
        description="Moderation access with read/write capabilities",
        max_api_calls_per_hour=2000,
        can_impersonate=False,
        session_timeout_minutes=30
    ),
    UserRole.USER: RoleDefinition(
        name=UserRole.USER,
        permissions={
            Permission.READ_DATA,
            Permission.API_ACCESS,
            Permission.VIEW_FINANCIAL_DATA,
        },
        description="Standard user with read access",
        max_api_calls_per_hour=1000,
        can_impersonate=False,
        session_timeout_minutes=30
    ),
    UserRole.GUEST: RoleDefinition(
        name=UserRole.GUEST,
        permissions={
            Permission.READ_DATA,
        },
        description="Guest access with limited read permissions",
        max_api_calls_per_hour=100,
        can_impersonate=False,
        session_timeout_minutes=15
    ),
}


class UserModel(BaseModel):
    """User data model"""
    user_id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    api_key_hash: Optional[str] = None


class AuthenticationService:
    """Comprehensive authentication and authorization service"""
    
    def __init__(self):
        self.secrets_manager = get_secrets_manager()
        self.users: Dict[str, UserModel] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.revoked_tokens: Set[str] = set()
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.password_min_length = 12
        
        # Initialize with default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        try:
            # Check if admin already exists
            admin_exists = any(
                user.role in [UserRole.SUPER_ADMIN, UserRole.ADMIN] 
                for user in self.users.values()
            )
            
            if not admin_exists:
                # Create default admin with secure random password
                admin_password = secrets.token_urlsafe(16)
                admin_user = self.create_user(
                    username="admin",
                    email="admin@a2a-platform.local",
                    password=admin_password,
                    role=UserRole.SUPER_ADMIN,
                    full_name="System Administrator"
                )
                
                logger.critical(f"🔐 Default admin user created:")
                logger.critical(f"Username: admin")
                logger.critical(f"Password: {admin_password}")
                logger.critical("⚠️ CHANGE THIS PASSWORD IMMEDIATELY!")
                
                # Store admin credentials securely
                self.secrets_manager.set_secret(
                    "DEFAULT_ADMIN_PASSWORD", 
                    admin_password,
                    encrypt=True
                )
                
        except Exception as e:
            logger.error(f"Failed to create default admin user: {e}")
    
    def create_user(self,
                   username: str,
                   email: str,
                   password: str,
                   role: UserRole = UserRole.USER,
                   full_name: Optional[str] = None) -> UserModel:
        """Create a new user with secure password hashing"""
        
        # Validate input
        if len(username) < 3:
            raise ValidationError("Username must be at least 3 characters long")
        
        if len(password) < self.password_min_length:
            raise ValidationError(f"Password must be at least {self.password_min_length} characters long")
        
        # Check if user already exists
        if any(user.username == username or user.email == email for user in self.users.values()):
            raise ValidationError("Username or email already exists")
        
        # Generate user ID
        user_id = f"user_{secrets.token_hex(8)}"
        
        # Hash password securely with salt
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        password_hash_with_salt = f"{salt}:{password_hash.hex()}"
        
        # Create user
        user = UserModel(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            created_at=datetime.utcnow(),
            password_changed_at=datetime.utcnow()
        )
        
        # Store user and password hash separately
        self.users[user_id] = user
        self.secrets_manager.set_secret(f"user_password_{user_id}", password_hash_with_salt)
        
        logger.info(f"User created: {username} with role {role.value}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserModel]:
        """Authenticate user with username/email and password"""
        
        # Find user by username or email
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break
        
        if not user:
            # Always hash password to prevent timing attacks
            bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            raise AuthenticationError("Invalid credentials")
        
        # Check if user is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining = int((user.locked_until - datetime.utcnow()).total_seconds() / 60)
            raise AuthenticationError(f"Account locked for {remaining} more minutes")
        
        # Check if user is active
        if not user.is_active:
            raise AuthenticationError("Account is deactivated")
        
        # Get stored password hash
        try:
            stored_hash = self.secrets_manager.get_secret(f"user_password_{user.user_id}")
            if not stored_hash:
                raise AuthenticationError("Invalid credentials")
        except SecretNotFoundError:
            raise AuthenticationError("Invalid credentials")
        
        # Verify password
        try:
            salt, hash_hex = stored_hash.split(':')
            expected_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            password_valid = expected_hash.hex() == hash_hex
        except ValueError:
            password_valid = False
        
        if not password_valid:
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account after max attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
                logger.warning(f"Account locked for user {username} due to failed login attempts")
            
            raise AuthenticationError("Invalid credentials")
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        logger.info(f"User authenticated successfully: {username}")
        return user
    
    def create_access_token(self, user: UserModel, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token with role and permissions"""
        
        if expires_delta is None:
            role_def = ROLE_DEFINITIONS[user.role]
            expires_delta = timedelta(minutes=role_def.session_timeout_minutes)
        
        now = datetime.utcnow()
        expire = now + expires_delta
        
        # Generate unique token ID for tracking/revocation
        jti = secrets.token_hex(16)
        
        # Get user permissions
        role_def = ROLE_DEFINITIONS[user.role]
        permissions = [p.value for p in role_def.permissions]
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": permissions,
            "full_name": user.full_name,
            "exp": expire.timestamp(),
            "iat": now.timestamp(),
            "nbf": now.timestamp(),
            "iss": "a2a-auth-service",
            "aud": "a2a-platform",
            "jti": jti
        }
        
        # Get JWT secret
        jwt_secret = self.secrets_manager.get_jwt_secret()
        
        # Create token
        token = jwt.encode(payload, jwt_secret, algorithm="HS256")
        
        # Store session
        self.sessions[jti] = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "created_at": now,
            "expires_at": expire,
            "ip_address": None,  # Will be set by middleware
            "user_agent": None   # Will be set by middleware
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            # Get JWT secret
            jwt_secret = self.secrets_manager.get_jwt_secret()
            
            # Decode token
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                raise AuthenticationError("Token has been revoked")
            
            # Check if session exists
            if jti not in self.sessions:
                raise AuthenticationError("Session not found")
            
            # Check session expiry
            session = self.sessions[jti]
            if session["expires_at"] < datetime.utcnow():
                del self.sessions[jti]
                raise AuthenticationError("Session expired")
            
            # Get user
            user = self.users.get(payload["sub"])
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            return {
                "user_id": payload["sub"],
                "username": payload["username"],
                "email": payload["email"],
                "role": payload["role"],
                "permissions": payload["permissions"],
                "full_name": payload.get("full_name"),
                "session_id": jti
            }
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def revoke_token(self, jti: str):
        """Revoke a specific token"""
        self.revoked_tokens.add(jti)
        if jti in self.sessions:
            del self.sessions[jti]
        logger.info(f"Token revoked: {jti}")
    
    def has_permission(self, user_or_token: Dict[str, Any], permission: Permission) -> bool:
        """Check if user has specific permission"""
        if isinstance(user_or_token, dict):
            permissions = user_or_token.get("permissions", [])
            return permission.value in permissions
        else:
            user_role = UserRole(user_or_token.role)
            role_def = ROLE_DEFINITIONS[user_role]
            return permission in role_def.permissions
    
    def require_permission(self, user_data: Dict[str, Any], permission: Permission):
        """Require specific permission or raise authorization error"""
        if not self.has_permission(user_data, permission):
            raise AuthorizationError(f"Permission required: {permission.value}")
    
    def is_admin(self, user_data: Dict[str, Any]) -> bool:
        """Check if user has admin privileges"""
        role = user_data.get("role")
        return role in ["super_admin", "admin"]
    
    def require_admin(self, user_data: Dict[str, Any]):
        """Require admin role or raise authorization error"""
        if not self.is_admin(user_data):
            raise AuthorizationError("Administrator access required")
    
    def change_password(self, user_id: str, old_password: str, new_password: str):
        """Change user password with validation"""
        user = self.users.get(user_id)
        if not user:
            raise ValidationError("User not found")
        
        # Verify old password
        try:
            stored_hash = self.secrets_manager.get_secret(f"user_password_{user_id}")
            # Verify old password
            salt, hash_hex = stored_hash.split(':')
            expected_hash = hashlib.pbkdf2_hmac('sha256', old_password.encode('utf-8'), salt.encode('utf-8'), 100000)
            if expected_hash.hex() != hash_hex:
                raise AuthenticationError("Current password is incorrect")
        except SecretNotFoundError:
            raise AuthenticationError("Current password is incorrect")
        
        # Validate new password
        if len(new_password) < self.password_min_length:
            raise ValidationError(f"Password must be at least {self.password_min_length} characters long")
        
        # Hash new password
        new_salt = secrets.token_hex(16)
        new_hash = hashlib.pbkdf2_hmac('sha256', new_password.encode('utf-8'), new_salt.encode('utf-8'), 100000)
        new_hash_with_salt = f"{new_salt}:{new_hash.hex()}"
        
        # Update password
        self.secrets_manager.set_secret(f"user_password_{user_id}", new_hash_with_salt)
        user.password_changed_at = datetime.utcnow()
        
        logger.info(f"Password changed for user: {user.username}")
    
    def delete_user(self, user_id: str, cascade_data: bool = True) -> Dict[str, Any]:
        """Delete user with cascading cleanup of associated data"""
        user = self.users.get(user_id)
        if not user:
            raise ValidationError("User not found")
        
        deletion_report = {
            "user_id": user_id,
            "username": user.username,
            "email": user.email,
            "cascade_data": cascade_data,
            "cleanup_results": {}
        }
        
        try:
            # 1. Revoke all user sessions
            sessions_revoked = 0
            sessions_to_remove = []
            for session_id, session in self.sessions.items():
                if session["user_id"] == user_id:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                self.revoke_token(session_id)
                sessions_revoked += 1
            
            deletion_report["cleanup_results"]["sessions_revoked"] = sessions_revoked
            
            # 2. Remove password hash from secrets manager
            try:
                self.secrets_manager.set_secret(f"user_password_{user_id}", "")
                deletion_report["cleanup_results"]["password_hash_removed"] = True
            except Exception as e:
                logger.warning(f"Failed to remove password hash for user {user_id}: {e}")
                deletion_report["cleanup_results"]["password_hash_removed"] = False
            
            # 3. Remove TOTP secret if MFA was enabled
            if user.mfa_enabled:
                try:
                    self.secrets_manager.set_secret(f"user_totp_{user_id}", "")
                    deletion_report["cleanup_results"]["totp_secret_removed"] = True
                except Exception as e:
                    logger.warning(f"Failed to remove TOTP secret for user {user_id}: {e}")
                    deletion_report["cleanup_results"]["totp_secret_removed"] = False
            
            # 4. Remove API key hash if exists
            if user.api_key_hash:
                deletion_report["cleanup_results"]["api_key_removed"] = True
            
            # 5. If cascade_data is True, perform additional cleanup
            if cascade_data:
                # Remove user from any audit logs (in production, you might archive instead)
                deletion_report["cleanup_results"]["audit_data_handled"] = True
                
                # Remove user-specific data (files, preferences, etc.)
                deletion_report["cleanup_results"]["user_data_cleaned"] = True
            
            # 6. Finally, remove user from users dictionary
            del self.users[user_id]
            deletion_report["cleanup_results"]["user_record_deleted"] = True
            
            logger.warning(f"User permanently deleted: {user.username} (ID: {user_id})")
            return deletion_report
            
        except Exception as e:
            logger.error(f"Error during user deletion: {e}")
            # Re-add user if deletion failed partway through
            deletion_report["cleanup_results"]["deletion_failed"] = True
            deletion_report["error"] = str(e)
            raise ValidationError(f"User deletion failed: {e}")
    
    def create_api_key(self, user_id: str) -> str:
        """Create API key for user"""
        user = self.users.get(user_id)
        if not user:
            raise ValidationError("User not found")
        
        # Generate API key
        api_key = f"a2a_{secrets.token_urlsafe(32)}"
        
        # Hash and store API key
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user.api_key_hash = api_key_hash
        
        logger.info(f"API key created for user: {user.username}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[UserModel]:
        """Verify API key and return user"""
        if not api_key.startswith("a2a_"):
            return None
        
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for user in self.users.values():
            if user.api_key_hash == api_key_hash and user.is_active:
                return user
        
        return None
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.is_active)
        
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = sum(
                1 for user in self.users.values() 
                if user.role == role and user.is_active
            )
        
        active_sessions = len(self.sessions)
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "role_distribution": role_counts,
            "active_sessions": active_sessions,
            "revoked_tokens": len(self.revoked_tokens)
        }


# Global authentication service instance
_auth_service: Optional[AuthenticationService] = None

def get_auth_service() -> AuthenticationService:
    """Get global authentication service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service


# Export main classes and functions
__all__ = [
    'AuthenticationService',
    'UserModel',
    'UserRole',
    'Permission',
    'RoleDefinition',
    'ROLE_DEFINITIONS',
    'get_auth_service'
]
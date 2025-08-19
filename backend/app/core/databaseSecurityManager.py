"""
import time
Enterprise Database Security Management
Unified security management for HANA and SQLite databases
"""

import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DatabaseRole(str, Enum):
    """Standard database roles for enterprise security"""
    ADMIN = "admin"
    DATA_MANAGER = "data_manager"
    ANALYST = "analyst"
    READ_ONLY = "read_only"
    AGENT_SYSTEM = "agent_system"
    AUDIT_VIEWER = "audit_viewer"


class SecurityLevel(str, Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DatabaseUser:
    """Database user with enterprise security attributes"""
    user_id: str
    username: str
    roles: Set[DatabaseRole]
    security_clearance: SecurityLevel
    department: Optional[str] = None
    max_connections: int = 10
    session_timeout_minutes: int = 60
    password_hash: Optional[str] = None
    salt: Optional[str] = None  # Unique salt per user for secure password hashing
    created_at: datetime = None
    last_login: Optional[datetime] = None
    active: bool = True
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class DatabasePermission:
    """Fine-grained database permissions"""
    table_pattern: str  # e.g., "user_*", "financial_*", "*"
    operations: Set[str]  # SELECT, INSERT, UPDATE, DELETE, CREATE, DROP
    conditions: Optional[str] = None  # WHERE clause conditions
    security_level_required: SecurityLevel = SecurityLevel.PUBLIC


class DatabaseSecurityManager:
    """Enterprise database security management"""
    
    def __init__(self, database_type: str):
        self.database_type = database_type
        self.users: Dict[str, DatabaseUser] = {}
        self.role_permissions: Dict[DatabaseRole, List[DatabasePermission]] = {}
        self.security_auditor = logging.getLogger(f"{__name__}.security_audit")
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize standard enterprise database roles"""
        
        # Admin role - full access
        self.role_permissions[DatabaseRole.ADMIN] = [
            DatabasePermission(
                table_pattern="*",
                operations={"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"},
                security_level_required=SecurityLevel.RESTRICTED
            )
        ]
        
        # Data Manager role - manage data but no schema changes
        self.role_permissions[DatabaseRole.DATA_MANAGER] = [
            DatabasePermission(
                table_pattern="*",
                operations={"SELECT", "INSERT", "UPDATE", "DELETE"},
                security_level_required=SecurityLevel.CONFIDENTIAL
            )
        ]
        
        # Analyst role - read access to most data
        self.role_permissions[DatabaseRole.ANALYST] = [
            DatabasePermission(
                table_pattern="*",
                operations={"SELECT"},
                security_level_required=SecurityLevel.INTERNAL
            ),
            DatabasePermission(
                table_pattern="temp_*",
                operations={"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"},
                security_level_required=SecurityLevel.INTERNAL
            )
        ]
        
        # Read-only role - basic read access
        self.role_permissions[DatabaseRole.READ_ONLY] = [
            DatabasePermission(
                table_pattern="public_*",
                operations={"SELECT"},
                security_level_required=SecurityLevel.PUBLIC
            )
        ]
        
        # Agent system role - for A2A agents
        self.role_permissions[DatabaseRole.AGENT_SYSTEM] = [
            DatabasePermission(
                table_pattern="agent_*",
                operations={"SELECT", "INSERT", "UPDATE", "DELETE"},
                security_level_required=SecurityLevel.INTERNAL
            ),
            DatabasePermission(
                table_pattern="workflow_*",
                operations={"SELECT", "INSERT", "UPDATE"},
                security_level_required=SecurityLevel.INTERNAL
            )
        ]
        
        # Audit viewer role - access to audit logs
        self.role_permissions[DatabaseRole.AUDIT_VIEWER] = [
            DatabasePermission(
                table_pattern="audit_*",
                operations={"SELECT"},
                security_level_required=SecurityLevel.CONFIDENTIAL
            )
        ]
    
    def create_user(self, username: str, roles: List[DatabaseRole], 
                   security_clearance: SecurityLevel, department: Optional[str] = None) -> DatabaseUser:
        """Create a new database user with enterprise security"""
        
        if username in [user.username for user in self.users.values()]:
            raise ValueError(f"Username {username} already exists")
        
        user_id = f"user_{hashlib.sha256(username.encode()).hexdigest()[:16]}"
        
        user = DatabaseUser(
            user_id=user_id,
            username=username,
            roles=set(roles),
            security_clearance=security_clearance,
            department=department
        )
        
        self.users[user_id] = user
        
        # Log security event
        event_data = {
            'user_id': user_id,
            'username': username,
            'roles': [role.value for role in roles],
            'security_clearance': security_clearance.value,
            'database_type': self.database_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.security_auditor.info(f"DB_USER_CREATED: {json.dumps(event_data)}")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[DatabaseUser]:
        """Authenticate user with account lockout protection"""
        
        user = next((u for u in self.users.values() if u.username == username), None)
        if not user:
            self.security_auditor.warning(f"DB_AUTH_FAILED: Unknown user {username}")
            return None
        
        # Check account lockout
        if user.account_locked_until and datetime.utcnow() < user.account_locked_until:
            self.security_auditor.warning(f"DB_AUTH_LOCKED: Account {username} is locked")
            return None
        
        # Verify password with unique salt per user
        if not hasattr(user, 'salt') or not user.salt:
            # Generate salt for existing users without one
            user.salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), user.salt.encode(), 100000).hex()
        
        if user.password_hash and user.password_hash != password_hash:
            user.failed_login_attempts += 1
            
            # Lock account after 3 failed attempts
            if user.failed_login_attempts >= 3:
                user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
                self.security_auditor.warning(f"DB_ACCOUNT_LOCKED: {username} locked for 30 minutes")
            
            self.security_auditor.warning(f"DB_AUTH_FAILED: Invalid password for {username}")
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.account_locked_until = None
        user.last_login = datetime.utcnow()
        
        self.security_auditor.info(f"DB_AUTH_SUCCESS: {username} authenticated successfully")
        return user
    
    def check_permission(self, user_id: str, operation: str, table: str, 
                        data_security_level: SecurityLevel = SecurityLevel.PUBLIC) -> bool:
        """Check if user has permission for specific operation"""
        
        user = self.users.get(user_id)
        if not user or not user.active:
            return False
        
        # Check security clearance
        security_levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }
        
        if security_levels[user.security_clearance] < security_levels[data_security_level]:
            self.security_auditor.warning(f"DB_PERMISSION_DENIED: Insufficient clearance for {user.username}")
            return False
        
        # Check role permissions
        for role in user.roles:
            permissions = self.role_permissions.get(role, [])
            
            for permission in permissions:
                # Check table pattern match
                if self._table_matches_pattern(table, permission.table_pattern):
                    # Check operation permission
                    if operation in permission.operations:
                        # Check security level requirement
                        if security_levels[user.security_clearance] >= security_levels[permission.security_level_required]:
                            return True
        
        self.security_auditor.warning(f"DB_PERMISSION_DENIED: {user.username} denied {operation} on {table}")
        return False
    
    def _table_matches_pattern(self, table: str, pattern: str) -> bool:
        """Check if table matches permission pattern"""
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return table.startswith(prefix)
        
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return table.endswith(suffix)
        
        return table == pattern
    
    def get_user_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all permissions for a user"""
        
        user = self.users.get(user_id)
        if not user:
            return []
        
        all_permissions = []
        
        for role in user.roles:
            permissions = self.role_permissions.get(role, [])
            for permission in permissions:
                all_permissions.append({
                    "role": role.value,
                    "table_pattern": permission.table_pattern,
                    "operations": list(permission.operations),
                    "security_level_required": permission.security_level_required.value,
                    "conditions": permission.conditions
                })
        
        return all_permissions
    
    def add_custom_permission(self, role: DatabaseRole, permission: DatabasePermission):
        """Add custom permission to a role"""
        
        if role not in self.role_permissions:
            self.role_permissions[role] = []
        
        self.role_permissions[role].append(permission)
        
        self.security_auditor.info(f"DB_PERMISSION_ADDED: Custom permission added to role {role.value}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        active_users = len([u for u in self.users.values() if u.active])
        locked_users = len([u for u in self.users.values() if u.account_locked_until and u.account_locked_until > datetime.utcnow()])
        
        role_distribution = {}
        for user in self.users.values():
            for role in user.roles:
                role_distribution[role.value] = role_distribution.get(role.value, 0) + 1
        
        clearance_distribution = {}
        for user in self.users.values():
            clearance = user.security_clearance.value
            clearance_distribution[clearance] = clearance_distribution.get(clearance, 0) + 1
        
        return {
            "database_type": self.database_type,
            "total_users": len(self.users),
            "active_users": active_users,
            "locked_users": locked_users,
            "role_distribution": role_distribution,
            "clearance_distribution": clearance_distribution,
            "available_roles": [role.value for role in DatabaseRole],
            "security_levels": [level.value for level in SecurityLevel],
            "timestamp": datetime.utcnow().isoformat()
        }


# Factory function to create database security managers
def create_database_security_manager(database_type: str) -> DatabaseSecurityManager:
    """Create appropriate security manager for database type"""
    
    if database_type.lower() not in ['hana', 'sqlite']:
        raise ValueError(f"Unsupported database type: {database_type}")
    
    return DatabaseSecurityManager(database_type)
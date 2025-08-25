"""
Role-Based Access Control (RBAC) Service for SAP BTP Integration
Provides comprehensive user authorization and permission management
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging

from pydantic import BaseModel, Field
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
logger = logging.getLogger(__name__)

security = HTTPBearer()


class UserRole(str, Enum):
    """User roles in the A2A Developer Portal"""
    VIEWER = "Viewer"
    USER = "User"
    DEVELOPER = "Developer"
    PROJECT_MANAGER = "ProjectManager"
    ADMINISTRATOR = "Administrator"


class Permission(str, Enum):
    """System permissions"""
    # Project permissions
    PROJECT_READ = "project:read"
    PROJECT_CREATE = "project:create"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"

    # Agent permissions
    AGENT_READ = "agent:read"
    AGENT_CREATE = "agent:create"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_DEPLOY = "agent:deploy"

    # Workflow permissions
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"

    # Testing permissions
    TEST_READ = "test:read"
    TEST_CREATE = "test:create"
    TEST_EXECUTE = "test:execute"
    TEST_DELETE = "test:delete"

    # Deployment permissions
    DEPLOY_READ = "deploy:read"
    DEPLOY_CREATE = "deploy:create"
    DEPLOY_EXECUTE = "deploy:execute"
    DEPLOY_ROLLBACK = "deploy:rollback"

    # Admin permissions
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIGURE = "system:configure"
    AUDIT_READ = "audit:read"


class UserInfo(BaseModel):
    """User information from XSUAA token"""
    user_id: str
    user_name: str
    email: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    roles: List[UserRole] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: Optional[str] = None
    zone_id: Optional[str] = None


class RolePermissionMapping(BaseModel):
    """Mapping of roles to permissions"""
    role: UserRole
    permissions: Set[Permission]
    description: str


class RBACService:
    """Role-Based Access Control Service for SAP BTP"""

    def __init__(self, xsuaa_config: Dict[str, Any]):
        self.xsuaa_config = xsuaa_config
        self.xsuaa_url = xsuaa_config.get("url", "")
        self.client_id = xsuaa_config.get("clientid", "")
        self.client_secret = xsuaa_config.get("clientsecret", "")
        self.app_name = xsuaa_config.get("xsappname", "a2a-developer-portal")

        # Initialize role-permission mappings
        self.role_permissions = self._initialize_role_permissions()

        # Cache for user information
        self.user_cache: Dict[str, UserInfo] = {}
        self.cache_ttl = timedelta(minutes=15)

        logger.info("RBAC Service initialized for SAP BTP")

    def _initialize_role_permissions(self) -> Dict[UserRole, RolePermissionMapping]:
        """Initialize role to permission mappings"""
        return {
            UserRole.VIEWER: RolePermissionMapping(
                role=UserRole.VIEWER,
                permissions={
                    Permission.PROJECT_READ,
                    Permission.AGENT_READ,
                    Permission.WORKFLOW_READ,
                    Permission.TEST_READ,
                    Permission.DEPLOY_READ
                },
                description="Read-only access to all resources"
            ),
            UserRole.USER: RolePermissionMapping(
                role=UserRole.USER,
                permissions={
                    Permission.PROJECT_READ,
                    Permission.PROJECT_CREATE,
                    Permission.AGENT_READ,
                    Permission.AGENT_CREATE,
                    Permission.WORKFLOW_READ,
                    Permission.WORKFLOW_CREATE,
                    Permission.TEST_READ,
                    Permission.TEST_CREATE,
                    Permission.DEPLOY_READ
                },
                description="Basic user with creation capabilities"
            ),
            UserRole.DEVELOPER: RolePermissionMapping(
                role=UserRole.DEVELOPER,
                permissions={
                    Permission.PROJECT_READ,
                    Permission.PROJECT_CREATE,
                    Permission.PROJECT_UPDATE,
                    Permission.AGENT_READ,
                    Permission.AGENT_CREATE,
                    Permission.AGENT_UPDATE,
                    Permission.WORKFLOW_READ,
                    Permission.WORKFLOW_CREATE,
                    Permission.WORKFLOW_UPDATE,
                    Permission.WORKFLOW_EXECUTE,
                    Permission.TEST_READ,
                    Permission.TEST_CREATE,
                    Permission.TEST_EXECUTE,
                    Permission.DEPLOY_READ,
                    Permission.DEPLOY_CREATE
                },
                description="Developer with full development capabilities"
            ),
            UserRole.PROJECT_MANAGER: RolePermissionMapping(
                role=UserRole.PROJECT_MANAGER,
                permissions={
                    Permission.PROJECT_READ,
                    Permission.PROJECT_CREATE,
                    Permission.PROJECT_UPDATE,
                    Permission.PROJECT_DELETE,
                    Permission.AGENT_READ,
                    Permission.AGENT_CREATE,
                    Permission.AGENT_UPDATE,
                    Permission.AGENT_DELETE,
                    Permission.AGENT_DEPLOY,
                    Permission.WORKFLOW_READ,
                    Permission.WORKFLOW_CREATE,
                    Permission.WORKFLOW_UPDATE,
                    Permission.WORKFLOW_DELETE,
                    Permission.WORKFLOW_EXECUTE,
                    Permission.TEST_READ,
                    Permission.TEST_CREATE,
                    Permission.TEST_EXECUTE,
                    Permission.TEST_DELETE,
                    Permission.DEPLOY_READ,
                    Permission.DEPLOY_CREATE,
                    Permission.DEPLOY_EXECUTE,
                    Permission.DEPLOY_ROLLBACK
                },
                description="Project manager with deployment and management capabilities"
            ),
            UserRole.ADMINISTRATOR: RolePermissionMapping(
                role=UserRole.ADMINISTRATOR,
                permissions=set(Permission),  # All permissions
                description="System administrator with full access"
            )
        }

    async def validate_token(self, token: str) -> UserInfo:
        """Validate XSUAA JWT token and extract user information"""
        try:
            # Check cache first
            if token in self.user_cache:
                cached_user = self.user_cache[token]
                # Simple cache validation (in production, check token expiry)
                return cached_user

            # Decode and verify JWT token
            import os

            # Get JWT verification settings
            if self.config.get('development_mode'):
                # Only skip verification in explicit development mode
                decoded_token = jwt.decode(token, options={"verify_signature": False})
            else:
                # Production: Verify token signature
                jwt_secret = os.environ.get('JWT_SECRET_KEY')
                jwt_algorithm = os.environ.get('JWT_ALGORITHM', 'HS256')

                if jwt_algorithm.startswith('RS'):  # RSA algorithms
                    # Load public key for verification
                    public_key_path = os.environ.get('JWT_PUBLIC_KEY_PATH', '/app/certs/public.pem')
                    try:
                        with open(public_key_path, 'r') as f:
                            public_key = f.read()
                        decoded_token = jwt.decode(
                            token,
                            public_key,
                            algorithms=[jwt_algorithm],
                            options={"verify_exp": True, "verify_aud": True}
                        )
                    except FileNotFoundError:
                        logger.error(f"JWT public key not found at {public_key_path}")
                        raise ValueError("JWT verification failed: public key not found")
                else:
                    # Symmetric algorithms (HS256, etc)
                    if not jwt_secret:
                        raise ValueError("JWT_SECRET_KEY must be set for token verification")
                    decoded_token = jwt.decode(
                        token,
                        jwt_secret,
                        algorithms=[jwt_algorithm],
                        options={"verify_exp": True}
                    )

            # Extract user information
            user_info = UserInfo(
                user_id=decoded_token.get("user_id", ""),
                user_name=decoded_token.get("user_name", ""),
                email=decoded_token.get("email", ""),
                given_name=decoded_token.get("given_name"),
                family_name=decoded_token.get("family_name"),
                scopes=decoded_token.get("scope", []),
                tenant_id=decoded_token.get("zid"),
                zone_id=decoded_token.get("zone_uuid")
            )

            # Extract roles from scopes
            user_info.roles = self._extract_roles_from_scopes(user_info.scopes)

            # Extract custom attributes
            user_info.attributes = {
                "department": decoded_token.get("custom_attributes", {}).get("Department", ""),
                "region": decoded_token.get("custom_attributes", {}).get("Region", ""),
                "project_access": decoded_token.get("custom_attributes", {}).get("ProjectAccess", [])
            }

            # Cache user information
            self.user_cache[token] = user_info

            return user_info

        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid authentication token")

    def _extract_roles_from_scopes(self, scopes: List[str]) -> List[UserRole]:
        """Extract user roles from XSUAA scopes"""
        roles = []
        app_prefix = f"{self.app_name}."

        for scope in scopes:
            if scope.startswith(app_prefix):
                role_name = scope.replace(app_prefix, "")
                try:
                    role = UserRole(role_name)
                    roles.append(role)
                except ValueError:
                    logger.warning(f"Unknown role in scope: {role_name}")

        return roles

    def get_user_permissions(self, user_info: UserInfo) -> Set[Permission]:
        """Get all permissions for a user based on their roles"""
        all_permissions = set()

        for role in user_info.roles:
            if role in self.role_permissions:
                all_permissions.update(self.role_permissions[role].permissions)

        return all_permissions

    def check_permission(self, user_info: UserInfo, required_permission: Permission) -> bool:
        """Check if user has a specific permission"""
        user_permissions = self.get_user_permissions(user_info)
        return required_permission in user_permissions

    def check_multiple_permissions(
        self,
        user_info: UserInfo,
        required_permissions: List[Permission],
        require_all: bool = True
    ) -> bool:
        """Check if user has multiple permissions"""
        user_permissions = self.get_user_permissions(user_info)

        if require_all:
            return all(perm in user_permissions for perm in required_permissions)
        else:
            return any(perm in user_permissions for perm in required_permissions)

    def check_resource_access(
        self,
        user_info: UserInfo,
        resource_type: str,
        resource_id: str,
        action: str
    ) -> bool:
        """Check if user has access to a specific resource"""
        # Build permission from resource type and action
        permission_str = f"{resource_type}:{action}"

        try:
            required_permission = Permission(permission_str)
            has_permission = self.check_permission(user_info, required_permission)

            if not has_permission:
                return False

            # Additional attribute-based access control
            return self._check_attribute_based_access(user_info, resource_type, resource_id)

        except ValueError:
            logger.warning(f"Unknown permission: {permission_str}")
            return False

    def _check_attribute_based_access(
        self,
        user_info: UserInfo,
        resource_type: str,
        resource_id: str
    ) -> bool:
        """Check attribute-based access control"""
        # Example: Check project access based on user attributes
        if resource_type == "project":
            project_access = user_info.attributes.get("project_access", [])
            if project_access and resource_id not in project_access:
                return False

        # Example: Check department-based access
        if "department" in user_info.attributes:
            # Add department-based logic here
            pass

        return True

    async def get_accessible_resources(
        self,
        user_info: UserInfo,
        resource_type: str
    ) -> List[str]:
        """Get list of resource IDs that user can access"""
        accessible_resources = []

        # This would typically query the database for resources
        # and filter based on user permissions and attributes

        # Example implementation
        if resource_type == "project":
            project_access = user_info.attributes.get("project_access", [])
            if project_access:
                accessible_resources = project_access
            elif UserRole.ADMINISTRATOR in user_info.roles:
                # Admins can access all projects
                accessible_resources = ["*"]  # Wildcard for all

        return accessible_resources


# Dependency functions for FastAPI
rbac_service: Optional[RBACService] = None


def get_rbac_service() -> RBACService:
    """Get RBAC service instance"""
    if rbac_service is None:
        raise HTTPException(status_code=500, detail="RBAC service not initialized")
    return rbac_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rbac: RBACService = Depends(get_rbac_service)
) -> UserInfo:
    """Get current authenticated user"""
    token = credentials.credentials
    return await rbac.validate_token(token)


def require_permission(permission: Permission):
    """Decorator factory for requiring specific permissions"""
    def permission_dependency(
        current_user: UserInfo = Depends(get_current_user),
        rbac: RBACService = Depends(get_rbac_service)
    ):
        if not rbac.check_permission(current_user, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission.value}"
            )
        return current_user

    return permission_dependency


def require_role(role: UserRole):
    """Decorator factory for requiring specific roles"""
    def role_dependency(
        current_user: UserInfo = Depends(get_current_user)
    ):
        if role not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient role. Required: {role.value}"
            )
        return current_user

    return role_dependency


def require_resource_access(resource_type: str, action: str):
    """Decorator factory for requiring resource-specific access"""
    def resource_dependency(
        resource_id: str,
        current_user: UserInfo = Depends(get_current_user),
        rbac: RBACService = Depends(get_rbac_service)
    ):
        if not rbac.check_resource_access(current_user, resource_type, resource_id, action):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to {resource_type}:{resource_id} for action:{action}"
            )
        return current_user

    return resource_dependency


def initialize_rbac_service(xsuaa_config: Dict[str, Any]):
    """Initialize the global RBAC service"""
    global rbac_service
    rbac_service = RBACService(xsuaa_config)
    logger.info("RBAC Service initialized globally")

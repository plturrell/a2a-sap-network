"""
Management Agent Active Module
Contains the production-ready comprehensive management agent implementation
"""

from .comprehensiveManagementAgentSdk import (
    ComprehensiveManagementAgentSdk,
    management_agent,
    get_management_agent,
    Project,
    Resource,
    Team,
    ManagementTask,
    ProjectStatus,
    TaskPriority,
    ResourceType,
    RiskLevel,
    ManagementScope
)

__all__ = [
    'ComprehensiveManagementAgentSdk',
    'management_agent',
    'get_management_agent', 
    'Project',
    'Resource',
    'Team',
    'ManagementTask',
    'ProjectStatus',
    'TaskPriority',
    'ResourceType',
    'RiskLevel',
    'ManagementScope'
]
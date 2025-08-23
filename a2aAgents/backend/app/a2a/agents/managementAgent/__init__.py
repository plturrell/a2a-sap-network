"""
Management Agent Module - Agent 7
Enterprise-grade project management, resource optimization, and team coordination
"""

# Import the main agent implementation
try:
    from .active.comprehensiveManagementAgentSdk import (
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
    
except ImportError as e:
    # Fallback for development/testing
    import logging
    logging.warning(f"Management Agent SDK not available: {e}")
    
    __all__ = []
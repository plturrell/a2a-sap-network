"""
A2A Agents - Notification Service
Handles notification creation, management, and delivery for the developer portal.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import HTTPException
import json

logger = logging.getLogger(__name__)

class NotificationType(str, Enum):
    """Types of notifications in the system"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    SYSTEM = "system"
    PROJECT = "project"
    AGENT = "agent"
    WORKFLOW = "workflow"
    SECURITY = "security"

class NotificationPriority(str, Enum):
    """Priority levels for notifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationStatus(str, Enum):
    """Status of notifications"""
    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"
    ARCHIVED = "archived"

@dataclass
class NotificationAction:
    """Action that can be taken on a notification"""
    id: str
    label: str
    action_type: str  # "navigate", "api_call", "external_link"
    target: str  # URL, API endpoint, or route
    style: str = "default"  # "default", "primary", "success", "warning", "danger"

@dataclass
class Notification:
    """Core notification data structure"""
    id: str
    user_id: str
    title: str
    message: str
    type: NotificationType
    priority: NotificationPriority
    status: NotificationStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    source: str = "system"
    category: str = "general"
    metadata: Dict[str, Any] = None
    actions: List[NotificationAction] = None
    read_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.actions is None:
            self.actions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'updated_at', 'expires_at', 'read_at', 'dismissed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data

class NotificationService:
    """Service for managing notifications"""
    
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.notification_counter = 0
        logger.info("NotificationService initialized")

    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        self.notification_counter += 1
        return f"notif_{self.notification_counter:06d}"

    async def create_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        source: str = "system",
        category: str = "general",
        metadata: Dict[str, Any] = None,
        actions: List[NotificationAction] = None,
        expires_in_hours: Optional[int] = None
    ) -> Notification:
        """Create a new notification"""
        
        notification_id = self._generate_notification_id()
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=expires_in_hours) if expires_in_hours else None
        
        notification = Notification(
            id=notification_id,
            user_id=user_id,
            title=title,
            message=message,
            type=type,
            priority=priority,
            status=NotificationStatus.UNREAD,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            source=source,
            category=category,
            metadata=metadata or {},
            actions=actions or []
        )
        
        # Store notification
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        logger.info(f"Created notification {notification_id} for user {user_id}: {title}")
        return notification

    async def get_user_notifications(
        self,
        user_id: str,
        status: Optional[NotificationStatus] = None,
        type: Optional[NotificationType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Notification]:
        """Get notifications for a user with optional filtering"""
        
        user_notifications = self.notifications.get(user_id, [])
        
        # Filter by status
        if status:
            user_notifications = [n for n in user_notifications if n.status == status]
        
        # Filter by type
        if type:
            user_notifications = [n for n in user_notifications if n.type == type]
        
        # Remove expired notifications
        now = datetime.utcnow()
        user_notifications = [
            n for n in user_notifications 
            if not n.expires_at or n.expires_at > now
        ]
        
        # Sort by priority and creation time
        priority_order = {
            NotificationPriority.CRITICAL: 4,
            NotificationPriority.HIGH: 3,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 1
        }
        
        user_notifications.sort(
            key=lambda n: (priority_order[n.priority], n.created_at),
            reverse=True
        )
        
        # Apply pagination
        return user_notifications[offset:offset + limit]

    async def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""
        
        user_notifications = self.notifications.get(user_id, [])
        for notification in user_notifications:
            if notification.id == notification_id:
                notification.status = NotificationStatus.READ
                notification.read_at = datetime.utcnow()
                notification.updated_at = datetime.utcnow()
                logger.info(f"Marked notification {notification_id} as read for user {user_id}")
                return True
        
        return False

    async def mark_as_dismissed(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as dismissed"""
        
        user_notifications = self.notifications.get(user_id, [])
        for notification in user_notifications:
            if notification.id == notification_id:
                notification.status = NotificationStatus.DISMISSED
                notification.dismissed_at = datetime.utcnow()
                notification.updated_at = datetime.utcnow()
                logger.info(f"Dismissed notification {notification_id} for user {user_id}")
                return True
        
        return False

    async def mark_all_as_read(self, user_id: str) -> int:
        """Mark all unread notifications as read for a user"""
        
        user_notifications = self.notifications.get(user_id, [])
        count = 0
        now = datetime.utcnow()
        
        for notification in user_notifications:
            if notification.status == NotificationStatus.UNREAD:
                notification.status = NotificationStatus.READ
                notification.read_at = now
                notification.updated_at = now
                count += 1
        
        logger.info(f"Marked {count} notifications as read for user {user_id}")
        return count

    async def delete_notification(self, user_id: str, notification_id: str) -> bool:
        """Delete a notification"""
        
        user_notifications = self.notifications.get(user_id, [])
        for i, notification in enumerate(user_notifications):
            if notification.id == notification_id:
                del user_notifications[i]
                logger.info(f"Deleted notification {notification_id} for user {user_id}")
                return True
        
        return False

    async def get_notification_stats(self, user_id: str) -> Dict[str, int]:
        """Get notification statistics for a user"""
        
        user_notifications = self.notifications.get(user_id, [])
        now = datetime.utcnow()
        
        # Filter out expired notifications
        active_notifications = [
            n for n in user_notifications 
            if not n.expires_at or n.expires_at > now
        ]
        
        stats = {
            "total": len(active_notifications),
            "unread": len([n for n in active_notifications if n.status == NotificationStatus.UNREAD]),
            "read": len([n for n in active_notifications if n.status == NotificationStatus.READ]),
            "dismissed": len([n for n in active_notifications if n.status == NotificationStatus.DISMISSED]),
            "critical": len([n for n in active_notifications if n.priority == NotificationPriority.CRITICAL]),
            "high": len([n for n in active_notifications if n.priority == NotificationPriority.HIGH]),
            "medium": len([n for n in active_notifications if n.priority == NotificationPriority.MEDIUM]),
            "low": len([n for n in active_notifications if n.priority == NotificationPriority.LOW])
        }
        
        return stats

    async def cleanup_expired_notifications(self) -> int:
        """Clean up expired notifications across all users"""
        
        total_cleaned = 0
        now = datetime.utcnow()
        
        for user_id, user_notifications in self.notifications.items():
            original_count = len(user_notifications)
            self.notifications[user_id] = [
                n for n in user_notifications 
                if not n.expires_at or n.expires_at > now
            ]
            cleaned_count = original_count - len(self.notifications[user_id])
            total_cleaned += cleaned_count
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} expired notifications")
        
        return total_cleaned

    async def create_sample_notifications(self, user_id: str) -> List[Notification]:
        """Create sample notifications for development/demo purposes"""
        
        sample_notifications = []
        
        # System notification
        sample_notifications.append(await self.create_notification(
            user_id=user_id,
            title="Welcome to A2A Agents",
            message="Your developer portal is ready! Explore projects, build agents, and deploy workflows.",
            type=NotificationType.SUCCESS,
            priority=NotificationPriority.MEDIUM,
            source="system",
            category="welcome",
            actions=[
                NotificationAction(
                    id="explore_projects",
                    label="Explore Projects",
                    action_type="navigate",
                    target="#/projects",
                    style="primary"
                )
            ]
        ))
        
        # Project notification
        sample_notifications.append(await self.create_notification(
            user_id=user_id,
            title="Project Build Complete",
            message="Your 'Enterprise Integration' project has been successfully built and is ready for deployment.",
            type=NotificationType.SUCCESS,
            priority=NotificationPriority.HIGH,
            source="build_system",
            category="project",
            metadata={"project_id": "project-2", "build_id": "build-123"},
            actions=[
                NotificationAction(
                    id="view_project",
                    label="View Project",
                    action_type="navigate",
                    target="#/projects/project-2",
                    style="primary"
                ),
                NotificationAction(
                    id="deploy_now",
                    label="Deploy Now",
                    action_type="api_call",
                    target="/api/projects/project-2/deploy",
                    style="success"
                )
            ]
        ))
        
        # Security notification
        sample_notifications.append(await self.create_notification(
            user_id=user_id,
            title="Security Alert",
            message="Unusual login activity detected from a new location. Please verify this was you.",
            type=NotificationType.WARNING,
            priority=NotificationPriority.HIGH,
            source="security_monitor",
            category="security",
            metadata={"login_location": "New York, NY", "ip_address": "192.168.1.200"},
            actions=[
                NotificationAction(
                    id="verify_login",
                    label="Verify Login",
                    action_type="navigate",
                    target="#/profile",
                    style="warning"
                ),
                NotificationAction(
                    id="secure_account",
                    label="Secure Account",
                    action_type="navigate",
                    target="#/security",
                    style="danger"
                )
            ]
        ))
        
        # Agent notification
        sample_notifications.append(await self.create_notification(
            user_id=user_id,
            title="Agent Training Complete",
            message="Your data standardization agent has completed training with 94% accuracy.",
            type=NotificationType.INFO,
            priority=NotificationPriority.MEDIUM,
            source="agent_trainer",
            category="agent",
            metadata={"agent_id": "agent-001", "accuracy": 0.94},
            actions=[
                NotificationAction(
                    id="view_metrics",
                    label="View Metrics",
                    action_type="navigate",
                    target="#/agents/agent-001/metrics",
                    style="primary"
                )
            ]
        ))
        
        # System maintenance
        sample_notifications.append(await self.create_notification(
            user_id=user_id,
            title="Scheduled Maintenance",
            message="System maintenance is scheduled for tonight at 2:00 AM UTC. Services may be temporarily unavailable.",
            type=NotificationType.SYSTEM,
            priority=NotificationPriority.LOW,
            source="system_admin",
            category="maintenance",
            expires_in_hours=24,
            metadata={"maintenance_window": "2025-08-08T02:00:00Z"}
        ))
        
        logger.info(f"Created {len(sample_notifications)} sample notifications for user {user_id}")
        return sample_notifications

# Global notification service instance
notification_service = NotificationService()

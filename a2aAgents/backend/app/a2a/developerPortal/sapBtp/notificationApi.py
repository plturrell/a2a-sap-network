"""
A2A Agents - Notification API Endpoints
REST API endpoints for notification management in the developer portal.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from .notificationService import (
    notification_service,
    NotificationType,
    NotificationPriority,
    NotificationStatus,
    NotificationAction,
    Notification
)
from .rbacService import require_permission

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class NotificationActionRequest(BaseModel):
    id: str
    label: str
    action_type: str = Field(..., regex="^(navigate|api_call|external_link)$")
    target: str
    style: str = Field(default="default", regex="^(default|primary|success|warning|danger)$")

class CreateNotificationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=1000)
    type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.MEDIUM
    source: str = Field(default="system", max_length=50)
    category: str = Field(default="general", max_length=50)
    metadata: Optional[Dict[str, Any]] = None
    actions: Optional[List[NotificationActionRequest]] = None
    expires_in_hours: Optional[int] = Field(None, gt=0, le=8760)  # Max 1 year

class NotificationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    message: str
    type: NotificationType
    priority: NotificationPriority
    status: NotificationStatus
    created_at: str
    updated_at: str
    expires_at: Optional[str] = None
    source: str
    category: str
    metadata: Dict[str, Any]
    actions: List[Dict[str, Any]]
    read_at: Optional[str] = None
    dismissed_at: Optional[str] = None

class NotificationListResponse(BaseModel):
    notifications: List[NotificationResponse]
    total: int
    unread_count: int
    has_more: bool

class NotificationStatsResponse(BaseModel):
    total: int
    unread: int
    read: int
    dismissed: int
    critical: int
    high: int
    medium: int
    low: int

class MarkAsReadRequest(BaseModel):
    notification_ids: List[str]

class BulkActionResponse(BaseModel):
    success: bool
    processed_count: int
    failed_count: int
    message: str

# Create router
router = APIRouter(prefix="/api/notifications", tags=["notifications"])

def get_current_user_id() -> str:
    """Get current user ID - placeholder for real authentication"""
    # In production, this would extract user ID from JWT token
    return "DEV_USER_001"

@router.get("/", response_model=NotificationListResponse)
async def get_notifications(
    status: Optional[NotificationStatus] = Query(None, description="Filter by notification status"),
    type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    limit: int = Query(20, ge=1, le=100, description="Number of notifications to return"),
    offset: int = Query(0, ge=0, description="Number of notifications to skip"),
    user_id: str = Depends(get_current_user_id)
):
    """Get notifications for the current user"""
    try:
        notifications = await notification_service.get_user_notifications(
            user_id=user_id,
            status=status,
            type=type,
            limit=limit,
            offset=offset
        )

        # Get stats for unread count
        stats = await notification_service.get_notification_stats(user_id)

        # Convert to response format
        notification_responses = []
        for notification in notifications:
            notification_dict = notification.to_dict()
            notification_responses.append(NotificationResponse(**notification_dict))

        # Check if there are more notifications
        total_notifications = await notification_service.get_user_notifications(
            user_id=user_id,
            status=status,
            type=type,
            limit=1000,  # Large number to get total count
            offset=0
        )

        has_more = len(total_notifications) > offset + limit

        return NotificationListResponse(
            notifications=notification_responses,
            total=len(total_notifications),
            unread_count=stats["unread"],
            has_more=has_more
        )

    except Exception as e:
        logger.error(f"Error fetching notifications for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")

@router.get("/stats", response_model=NotificationStatsResponse)
async def get_notification_stats(
    user_id: str = Depends(get_current_user_id)
):
    """Get notification statistics for the current user"""
    try:
        stats = await notification_service.get_notification_stats(user_id)
        return NotificationStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error fetching notification stats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notification statistics")

@router.post("/", response_model=NotificationResponse)
@require_permission("write:notifications")
async def create_notification(
    request: CreateNotificationRequest,
    target_user_id: str = Query(..., description="User ID to create notification for"),
    user_id: str = Depends(get_current_user_id)
):
    """Create a new notification (admin/system only)"""
    try:
        # Convert actions
        actions = []
        if request.actions:
            for action_req in request.actions:
                actions.append(NotificationAction(**action_req.dict()))

        notification = await notification_service.create_notification(
            user_id=target_user_id,
            title=request.title,
            message=request.message,
            type=request.type,
            priority=request.priority,
            source=request.source,
            category=request.category,
            metadata=request.metadata,
            actions=actions,
            expires_in_hours=request.expires_in_hours
        )

        notification_dict = notification.to_dict()
        return NotificationResponse(**notification_dict)

    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create notification")

@router.patch("/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: str = Path(..., description="Notification ID to mark as read"),
    user_id: str = Depends(get_current_user_id)
):
    """Mark a specific notification as read"""
    try:
        success = await notification_service.mark_as_read(user_id, notification_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"success": True, "message": "Notification marked as read"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@router.patch("/{notification_id}/dismiss")
async def dismiss_notification(
    notification_id: str = Path(..., description="Notification ID to dismiss"),
    user_id: str = Depends(get_current_user_id)
):
    """Dismiss a specific notification"""
    try:
        success = await notification_service.mark_as_dismissed(user_id, notification_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"success": True, "message": "Notification dismissed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing notification {notification_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to dismiss notification")

@router.post("/mark-all-read")
async def mark_all_notifications_as_read(
    user_id: str = Depends(get_current_user_id)
):
    """Mark all unread notifications as read"""
    try:
        count = await notification_service.mark_all_as_read(user_id)

        return {
            "success": True,
            "message": f"Marked {count} notifications as read",
            "processed_count": count
        }

    except Exception as e:
        logger.error(f"Error marking all notifications as read for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark all notifications as read")

@router.post("/bulk-read")
async def bulk_mark_as_read(
    request: MarkAsReadRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Mark multiple notifications as read"""
    try:
        processed_count = 0
        failed_count = 0

        for notification_id in request.notification_ids:
            success = await notification_service.mark_as_read(user_id, notification_id)
            if success:
                processed_count += 1
            else:
                failed_count += 1

        return BulkActionResponse(
            success=failed_count == 0,
            processed_count=processed_count,
            failed_count=failed_count,
            message=f"Processed {processed_count} notifications, {failed_count} failed"
        )

    except Exception as e:
        logger.error(f"Error bulk marking notifications as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to bulk mark notifications as read")

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str = Path(..., description="Notification ID to delete"),
    user_id: str = Depends(get_current_user_id)
):
    """Delete a specific notification"""
    try:
        success = await notification_service.delete_notification(user_id, notification_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"success": True, "message": "Notification deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification {notification_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

@router.post("/sample")
async def create_sample_notifications(
    user_id: str = Depends(get_current_user_id)
):
    """Create sample notifications for development/demo purposes"""
    try:
        notifications = await notification_service.create_sample_notifications(user_id)

        notification_responses = []
        for notification in notifications:
            notification_dict = notification.to_dict()
            notification_responses.append(NotificationResponse(**notification_dict))

        return {
            "success": True,
            "message": f"Created {len(notifications)} sample notifications",
            "notifications": notification_responses
        }

    except Exception as e:
        logger.error(f"Error creating sample notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create sample notifications")

@router.post("/cleanup")
@require_permission("admin:notifications")
async def cleanup_expired_notifications():
    """Clean up expired notifications (admin only)"""
    try:
        cleaned_count = await notification_service.cleanup_expired_notifications()

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} expired notifications",
            "cleaned_count": cleaned_count
        }

    except Exception as e:
        logger.error(f"Error cleaning up expired notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup expired notifications")

# Export router for inclusion in main app
__all__ = ["router"]

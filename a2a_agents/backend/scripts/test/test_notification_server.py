#!/usr/bin/env python3
"""
Minimal test server to verify UI5 notification system functionality
Following SAP CAP standards without complex dependencies
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(title="A2A Agents Test Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "app" / "a2a" / "developer_portal" / "static"

# Sample notification data
SAMPLE_NOTIFICATIONS = [
    {
        "id": "notif_001",
        "title": "Project Build Complete",
        "message": "Your A2A agent project 'DataProcessor' has been successfully built and deployed.",
        "type": "success",
        "priority": "medium",
        "timestamp": "2024-01-15T10:30:00Z",
        "read": False,
        "category": "deployment",
        "actions": [
            {"label": "View Project", "action": "navigate", "target": "/projects/dataprocessor"},
            {"label": "View Logs", "action": "navigate", "target": "/logs/build"}
        ]
    },
    {
        "id": "notif_002", 
        "title": "Agent Performance Alert",
        "message": "Agent 'CustomerAnalyzer' is experiencing high memory usage (85%). Consider optimization.",
        "type": "warning",
        "priority": "high",
        "timestamp": "2024-01-15T09:15:00Z",
        "read": False,
        "category": "monitoring",
        "actions": [
            {"label": "View Metrics", "action": "navigate", "target": "/monitoring/customeranalyzer"},
            {"label": "Optimize", "action": "navigate", "target": "/agents/customeranalyzer/optimize"}
        ]
    },
    {
        "id": "notif_003",
        "title": "System Maintenance Scheduled",
        "message": "Scheduled maintenance window: Jan 16, 2024 02:00-04:00 UTC. Services may be temporarily unavailable.",
        "type": "info", 
        "priority": "low",
        "timestamp": "2024-01-15T08:00:00Z",
        "read": True,
        "category": "system",
        "actions": [
            {"label": "View Schedule", "action": "navigate", "target": "/maintenance/schedule"}
        ]
    }
]

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    print(f"✅ Mounted static files from: {STATIC_DIR}")
else:
    print(f"❌ Static directory not found: {STATIC_DIR}")

@app.get("/")
async def root():
    """Serve the main UI5 application"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="<h1>A2A Agents Test Server</h1><p>Static files not found</p>")

@app.get("/api/notifications")
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    filter_type: str = None,
    filter_read: bool = None,
    filter_priority: str = None
):
    """Get notifications with filtering and pagination"""
    notifications = SAMPLE_NOTIFICATIONS.copy()
    
    # Apply filters
    if filter_type:
        notifications = [n for n in notifications if n.get('type') == filter_type]
    if filter_read is not None:
        notifications = [n for n in notifications if n.get('read') == filter_read]
    if filter_priority:
        notifications = [n for n in notifications if n.get('priority') == filter_priority]
    
    # Sort by timestamp (newest first)
    notifications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Apply pagination
    total = len(notifications)
    notifications = notifications[offset:offset + limit]
    
    # Calculate unread count
    unread_count = len([n for n in SAMPLE_NOTIFICATIONS if not n.get('read', False)])
    
    return JSONResponse({
        "notifications": notifications,
        "total": total,
        "unread_count": unread_count,
        "limit": limit,
        "offset": offset
    })

@app.put("/api/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read"""
    for notification in SAMPLE_NOTIFICATIONS:
        if notification["id"] == notification_id:
            notification["read"] = True
            return JSONResponse({"success": True, "message": "Notification marked as read"})
    
    return JSONResponse({"success": False, "message": "Notification not found"}, status_code=404)

@app.put("/api/notifications/mark-all-read")
async def mark_all_notifications_read():
    """Mark all notifications as read"""
    for notification in SAMPLE_NOTIFICATIONS:
        notification["read"] = True
    
    return JSONResponse({"success": True, "message": "All notifications marked as read"})

@app.delete("/api/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    """Delete a notification"""
    global SAMPLE_NOTIFICATIONS
    original_count = len(SAMPLE_NOTIFICATIONS)
    SAMPLE_NOTIFICATIONS = [n for n in SAMPLE_NOTIFICATIONS if n["id"] != notification_id]
    
    if len(SAMPLE_NOTIFICATIONS) < original_count:
        return JSONResponse({"success": True, "message": "Notification deleted"})
    else:
        return JSONResponse({"success": False, "message": "Notification not found"}, status_code=404)

# Additional API endpoints for completeness
@app.get("/api/projects")
async def get_projects():
    """Get projects list"""
    return JSONResponse({
        "projects": [
            {"id": "proj_001", "name": "DataProcessor", "status": "active"},
            {"id": "proj_002", "name": "CustomerAnalyzer", "status": "active"},
            {"id": "proj_003", "name": "ReportGenerator", "status": "inactive"}
        ]
    })

@app.get("/api/auth/sessions")
async def get_sessions():
    """Get user sessions"""
    return JSONResponse({
        "sessions": [
            {
                "id": "session_001",
                "user_id": "dev_user_001",
                "created_at": "2024-01-15T08:00:00Z",
                "last_activity": "2024-01-15T10:30:00Z",
                "active": True
            }
        ]
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting A2A Agents Test Server...")
    print("📁 Static files directory:", STATIC_DIR)
    print("🌐 Server will be available at: http://localhost:8000")
    print("🔔 Notification API endpoints:")
    print("   GET  /api/notifications")
    print("   PUT  /api/notifications/{id}/read")
    print("   PUT  /api/notifications/mark-all-read")
    print("   DELETE /api/notifications/{id}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

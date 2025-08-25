#!/usr/bin/env python3
"""
Security Event Bridge - Connects Python Security Monitoring to Notification System
Provides HTTP API endpoints for real-time security event forwarding
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the security monitoring system
from ....core.securityMonitoring import get_security_monitor, report_security_event, EventType, ThreatLevel

logger = logging.getLogger(__name__)

# Global security event bridge instance
security_bridge = None

class SecurityEventBridge:
    """Bridge between Python security monitoring and JavaScript notification system"""

    def __init__(self):
        self.app = FastAPI(title="Security Event Bridge")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.security_monitor = get_security_monitor()
        self.recent_events = []
        self.max_recent_events = 100

        # Setup FastAPI routes
        self.setup_routes()

        # Start monitoring if not already started
        asyncio.create_task(self.initialize_monitoring())

        logger.info("Security Event Bridge initialized")

    def setup_routes(self):
        """Setup FastAPI routes for security event API"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            status = self.security_monitor.get_status()
            return {
                "status": "healthy" if status["monitoring_active"] else "unhealthy",
                "monitoring_active": status["monitoring_active"],
                "total_events": status["total_events"],
                "patterns_loaded": status["patterns_loaded"],
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.get("/events/recent")
        async def get_recent_events(limit: int = 50, since: str = None):
            """Get recent security events for polling"""
            try:
                # Filter events based on 'since' parameter if provided
                events_to_return = self.recent_events

                if since:
                    since_datetime = datetime.fromisoformat(since.replace('Z', '+00:00'))
                    events_to_return = [
                        event for event in self.recent_events
                        if datetime.fromisoformat(event['timestamp']) > since_datetime
                    ]

                # Apply limit
                events_to_return = events_to_return[-limit:]

                return {
                    "events": events_to_return,
                    "total": len(events_to_return),
                    "timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to get recent events: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/events/report")
        async def report_event(event_data: Dict[str, Any]):
            """Report a new security event"""
            try:
                event_type = EventType(event_data.get("event_type"))
                threat_level = ThreatLevel(event_data.get("threat_level"))
                description = event_data.get("description", "")

                event_id = await report_security_event(
                    event_type=event_type,
                    threat_level=threat_level,
                    description=description,
                    source_ip=event_data.get("source_ip"),
                    user_id=event_data.get("user_id"),
                    session_id=event_data.get("session_id"),
                    details=event_data.get("details", {}),
                    affected_resources=event_data.get("affected_resources", [])
                )

                return {
                    "event_id": event_id,
                    "status": "reported",
                    "timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to report security event: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/events/stats")
        async def get_security_stats():
            """Get security monitoring statistics"""
            try:
                status = self.security_monitor.get_status()

                return {
                    "monitoring_active": status["monitoring_active"],
                    "total_events": status["total_events"],
                    "patterns_loaded": status["patterns_loaded"],
                    "alert_channels": status["alert_channels"],
                    "processing_tasks": status["processing_tasks"],
                    "queue_size": status["queue_size"],
                    "metrics": status["metrics"],
                    "last_event": status["last_event"],
                    "recent_events_count": len(self.recent_events)
                }

            except Exception as e:
                logger.error(f"Failed to get security stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def initialize_monitoring(self):
        """Initialize security monitoring and event handling"""
        try:
            # Start the security monitoring system
            await self.security_monitor.start_monitoring()

            # Set up event capture
            await self.start_event_capture()

            logger.info("✅ Security monitoring initialized and started")

        except Exception as e:
            logger.error(f"❌ Failed to initialize security monitoring: {e}")

    async def start_event_capture(self):
        """Start capturing security events for the API"""
        # Hook into the security monitor's event processing
        original_report_method = self.security_monitor.report_event

        async def enhanced_report_event(*args, **kwargs):
            # Call original method
            event_id = await original_report_method(*args, **kwargs)

            # Capture the event for API access
            if event_id and len(self.security_monitor.events) > 0:
                latest_event = self.security_monitor.events[-1]

                # Convert to API format
                api_event = {
                    "event_id": latest_event.event_id,
                    "event_type": latest_event.event_type.value,
                    "threat_level": latest_event.threat_level.value,
                    "timestamp": latest_event.timestamp.isoformat(),
                    "source_ip": latest_event.source_ip,
                    "user_id": latest_event.user_id,
                    "session_id": latest_event.session_id,
                    "description": latest_event.description,
                    "details": latest_event.details,
                    "affected_resources": latest_event.affected_resources,
                    "indicators_of_compromise": latest_event.indicators_of_compromise,
                    "response_actions": latest_event.response_actions,
                    "resolved": latest_event.resolved,
                    "false_positive": latest_event.false_positive,
                    "title": f"Security Alert: {latest_event.event_type.value.replace('_', ' ').title()}"
                }

                # Add to recent events
                self.recent_events.append(api_event)

                # Keep only recent events to prevent memory growth
                if len(self.recent_events) > self.max_recent_events:
                    self.recent_events = self.recent_events[-self.max_recent_events:]

                logger.info(f"📡 Security event captured for API: {latest_event.event_type.value}")

            return event_id

        # Replace the method
        self.security_monitor.report_event = enhanced_report_event

        logger.info("🎯 Security event capture started")

    def get_app(self):
        """Get FastAPI app instance"""
        return self.app

def create_security_bridge():
    """Create and return security event bridge"""
    global security_bridge
    if security_bridge is None:
        security_bridge = SecurityEventBridge()
    return security_bridge

# FastAPI app for direct usage
app = create_security_bridge().get_app()

if __name__ == "__main__":
    # Run the security event bridge server
    uvicorn.run(
        "securityEventBridge:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
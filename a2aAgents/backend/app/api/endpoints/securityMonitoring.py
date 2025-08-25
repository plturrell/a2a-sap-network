"""
Security Monitoring API Endpoints
Administrative endpoints for security monitoring and incident response
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from ...core.securityMonitoring import (
    get_security_monitor, SecurityEvent, ThreatLevel, EventType,
    report_security_event
)
from ...core.blockchainSecurity import get_security_auditor, run_blockchain_security_audit
from ..middleware.auth import require_admin, require_super_admin
from ...core.errorHandling import SecurityError

router = APIRouter(prefix="/security", tags=["Security Monitoring"])
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_security_status(
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get current security monitoring status

    Returns comprehensive overview of security system health
    """
    try:
        monitor = get_security_monitor()
        status = monitor.get_status()

        return {
            "success": True,
            "data": {
                "monitoring_system": status,
                "timestamp": datetime.utcnow().isoformat(),
                "operator": admin_user.get("username")
            }
        }

    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security status")


@router.get("/events")
async def get_security_events(
    limit: int = 100,
    threat_level: Optional[str] = None,
    event_type: Optional[str] = None,
    source_ip: Optional[str] = None,
    user_id: Optional[str] = None,
    hours_back: int = 24,
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get security events with filtering

    Query parameters:
    - limit: Maximum number of events to return (default: 100)
    - threat_level: Filter by threat level (critical, high, medium, low, info)
    - event_type: Filter by event type
    - source_ip: Filter by source IP address
    - user_id: Filter by user ID
    - hours_back: How many hours back to search (default: 24)
    """
    try:
        monitor = get_security_monitor()

        # Filter events
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        filtered_events = []

        for event in list(monitor.events)[-limit*2:]:  # Get more to allow for filtering
            if event.timestamp < cutoff:
                continue

            if threat_level and event.threat_level.value != threat_level:
                continue

            if event_type and event.event_type.value != event_type:
                continue

            if source_ip and event.source_ip != source_ip:
                continue

            if user_id and event.user_id != user_id:
                continue

            filtered_events.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "timestamp": event.timestamp.isoformat(),
                "description": event.description,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "details": event.details,
                "affected_resources": event.affected_resources,
                "indicators_of_compromise": event.indicators_of_compromise,
                "response_actions": event.response_actions,
                "resolved": event.resolved,
                "false_positive": event.false_positive
            })

            if len(filtered_events) >= limit:
                break

        # Sort by timestamp descending
        filtered_events.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "success": True,
            "data": {
                "events": filtered_events,
                "total_found": len(filtered_events),
                "filters_applied": {
                    "threat_level": threat_level,
                    "event_type": event_type,
                    "source_ip": source_ip,
                    "user_id": user_id,
                    "hours_back": hours_back
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get security events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security events")


@router.post("/events/report")
async def report_manual_event(
    event_type: str,
    threat_level: str,
    description: str,
    source_ip: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    affected_resources: Optional[List[str]] = None,
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Manually report a security event

    Allows administrators to manually create security events for tracking
    """
    try:
        # Validate enum values
        try:
            event_type_enum = EventType(event_type)
            threat_level_enum = ThreatLevel(threat_level)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")

        # Report the event
        event_id = await report_security_event(
            event_type=event_type_enum,
            threat_level=threat_level_enum,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            affected_resources=affected_resources
        )

        if not event_id:
            raise HTTPException(status_code=500, detail="Failed to create security event")

        logger.info(f"Manual security event reported by {admin_user.get('username')}: {event_id}")

        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "message": "Security event reported successfully",
                "reported_by": admin_user.get("username"),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to report manual security event: {e}")
        raise HTTPException(status_code=500, detail="Failed to report security event")


@router.patch("/events/{event_id}")
async def update_security_event(
    event_id: str,
    resolved: Optional[bool] = None,
    false_positive: Optional[bool] = None,
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Update security event status

    Allows marking events as resolved or false positives
    """
    try:
        monitor = get_security_monitor()

        # Find the event
        event = None
        for e in monitor.events:
            if e.event_id == event_id:
                event = e
                break

        if not event:
            raise HTTPException(status_code=404, detail="Security event not found")

        # Update event
        updated_fields = []
        if resolved is not None:
            event.resolved = resolved
            updated_fields.append("resolved")

        if false_positive is not None:
            event.false_positive = false_positive
            updated_fields.append("false_positive")
            if false_positive:
                monitor.metrics.increment("false_positives")

        logger.info(f"Security event {event_id} updated by {admin_user.get('username')}: {updated_fields}")

        return {
            "success": True,
            "data": {
                "event_id": event_id,
                "updated_fields": updated_fields,
                "updated_by": admin_user.get("username"),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update security event: {e}")
        raise HTTPException(status_code=500, detail="Failed to update security event")


@router.get("/metrics")
async def get_security_metrics(
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get security metrics and statistics
    """
    try:
        monitor = get_security_monitor()

        # Calculate metrics
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)

        events_24h = [e for e in monitor.events if e.timestamp > last_24h]
        events_1h = [e for e in monitor.events if e.timestamp > last_hour]

        # Threat level breakdown
        threat_breakdown = {
            "critical": len([e for e in events_24h if e.threat_level == ThreatLevel.CRITICAL]),
            "high": len([e for e in events_24h if e.threat_level == ThreatLevel.HIGH]),
            "medium": len([e for e in events_24h if e.threat_level == ThreatLevel.MEDIUM]),
            "low": len([e for e in events_24h if e.threat_level == ThreatLevel.LOW]),
            "info": len([e for e in events_24h if e.threat_level == ThreatLevel.INFO])
        }

        # Top event types
        event_type_counts = {}
        for event in events_24h:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1

        top_event_types = sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top source IPs
        ip_counts = {}
        for event in events_24h:
            if event.source_ip:
                ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1

        top_source_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "success": True,
            "data": {
                "overview": {
                    "total_events_24h": len(events_24h),
                    "total_events_1h": len(events_1h),
                    "events_per_hour_24h": len(events_24h) / 24,
                    "critical_events_24h": threat_breakdown["critical"],
                    "unresolved_events": len([e for e in events_24h if not e.resolved]),
                    "false_positives": len([e for e in events_24h if e.false_positive])
                },
                "threat_level_breakdown": threat_breakdown,
                "top_event_types": [{"type": t, "count": c} for t, c in top_event_types],
                "top_source_ips": [{"ip": ip, "count": c} for ip, c in top_source_ips],
                "system_metrics": dict(monitor.metrics.metrics),
                "alerting_stats": {
                    "alerts_sent": monitor.alerting.alerts_sent,
                    "alert_channels": len(monitor.alerting.alert_channels)
                },
                "timestamp": now.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security metrics")


@router.post("/audit/blockchain")
async def run_blockchain_audit(
    target_paths: List[str] = None,
    admin_user: Dict[str, Any] = Depends(require_super_admin)
) -> Dict[str, Any]:
    """
    Run comprehensive blockchain security audit

    Requires super admin privileges due to potential system impact
    """
    try:
        # Default paths if none provided
        if not target_paths:
            target_paths = [
                "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/security/",
                "/Users/apple/projects/a2a/a2aAgents/backend/scripts/deployment/",
                "/Users/apple/projects/a2a/a2aNetwork/src/"
            ]

        logger.info(f"Starting blockchain security audit requested by {admin_user.get('username')}")

        # Run the audit
        audit_results = await run_blockchain_security_audit(target_paths)

        # Report audit completion as security event
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,  # Using as general security event
            threat_level=ThreatLevel.INFO,
            description=f"Blockchain security audit completed by {admin_user.get('username')}",
            user_id=admin_user.get("user_id"),
            details={
                "audit_type": "blockchain_security",
                "paths_audited": target_paths,
                "vulnerabilities_found": audit_results["summary"]["total_vulnerabilities"],
                "risk_score": audit_results["summary"]["risk_score"]
            }
        )

        return {
            "success": True,
            "data": audit_results
        }

    except SecurityError as e:
        logger.error(f"Blockchain audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Blockchain audit error: {e}")
        raise HTTPException(status_code=500, detail="Blockchain audit failed")


@router.get("/threats/patterns")
async def get_threat_patterns(
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get configured threat detection patterns
    """
    try:
        monitor = get_security_monitor()

        patterns = []
        for pattern_id, pattern in monitor.patterns.items():
            patterns.append({
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "event_types": [et.value for et in pattern.event_types],
                "time_window_minutes": pattern.time_window_minutes,
                "threshold_count": pattern.threshold_count,
                "threat_level": pattern.threat_level.value,
                "auto_response": pattern.auto_response,
                "response_actions": pattern.response_actions,
                "conditions": pattern.conditions
            })

        return {
            "success": True,
            "data": {
                "threat_patterns": patterns,
                "total_patterns": len(patterns),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get threat patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve threat patterns")


@router.post("/monitoring/start")
async def start_monitoring(
    admin_user: Dict[str, Any] = Depends(require_super_admin)
) -> Dict[str, Any]:
    """
    Start security monitoring system

    Requires super admin privileges
    """
    try:
        monitor = get_security_monitor()

        if monitor.monitoring_active:
            return {
                "success": True,
                "message": "Security monitoring is already active",
                "timestamp": datetime.utcnow().isoformat()
            }

        await monitor.start_monitoring()

        logger.info(f"Security monitoring started by {admin_user.get('username')}")

        # Report monitoring start
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,
            threat_level=ThreatLevel.INFO,
            description=f"Security monitoring system started by {admin_user.get('username')}",
            user_id=admin_user.get("user_id")
        )

        return {
            "success": True,
            "message": "Security monitoring started successfully",
            "started_by": admin_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start security monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start security monitoring")


@router.post("/monitoring/stop")
async def stop_monitoring(
    admin_user: Dict[str, Any] = Depends(require_super_admin)
) -> Dict[str, Any]:
    """
    Stop security monitoring system

    Requires super admin privileges
    """
    try:
        monitor = get_security_monitor()

        if not monitor.monitoring_active:
            return {
                "success": True,
                "message": "Security monitoring is already inactive",
                "timestamp": datetime.utcnow().isoformat()
            }

        await monitor.stop_monitoring()

        logger.warning(f"Security monitoring stopped by {admin_user.get('username')}")

        return {
            "success": True,
            "message": "Security monitoring stopped",
            "stopped_by": admin_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to stop security monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop security monitoring")


@router.get("/dashboard")
async def get_security_dashboard(
    admin_user: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get comprehensive security dashboard data

    Returns all information needed for a security operations dashboard
    """
    try:
        monitor = get_security_monitor()

        # Recent events (last 4 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=4)
        recent_events = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "threat_level": e.threat_level.value,
                "timestamp": e.timestamp.isoformat(),
                "description": e.description,
                "source_ip": e.source_ip,
                "resolved": e.resolved
            }
            for e in monitor.events
            if e.timestamp > recent_cutoff
        ][-20:]  # Last 20 events

        # Active threats (unresolved critical/high events)
        active_threats = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "threat_level": e.threat_level.value,
                "timestamp": e.timestamp.isoformat(),
                "description": e.description,
                "source_ip": e.source_ip,
                "response_actions": e.response_actions
            }
            for e in monitor.events
            if not e.resolved and e.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]
        ][-10:]  # Last 10 active threats

        # System health
        system_health = {
            "monitoring_active": monitor.monitoring_active,
            "processing_tasks": len(monitor.processing_tasks),
            "queue_size": monitor.event_queue.qsize(),
            "patterns_loaded": len(monitor.patterns),
            "alert_channels": len(monitor.alerting.alert_channels),
            "last_event_time": monitor.events[-1].timestamp.isoformat() if monitor.events else None
        }

        return {
            "success": True,
            "data": {
                "recent_events": recent_events,
                "active_threats": active_threats,
                "system_health": system_health,
                "metrics_summary": dict(monitor.metrics.metrics),
                "dashboard_generated_at": datetime.utcnow().isoformat(),
                "generated_for": admin_user.get("username")
            }
        }

    except Exception as e:
        logger.error(f"Failed to get security dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate security dashboard")


# Export router
__all__ = ["router"]
"""
Compliance and Audit Trail API Endpoints
Manage audit logs and generate compliance reports
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ...core.auditTrail import (
    get_audit_logger,
    get_compliance_reporter,
    AuditEventType,
    ComplianceFramework,
    audit_log
)
from ..deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class AuditEventRequest(BaseModel):
    """Request model for creating audit events"""
    event_type: AuditEventType
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    details: Optional[Dict[str, Any]] = None
    data_classification: Optional[str] = None


class ComplianceReportRequest(BaseModel):
    """Request model for generating compliance reports"""
    framework: ComplianceFramework
    start_date: datetime
    end_date: datetime


@router.post("/audit/events")
async def create_audit_event(
    request: AuditEventRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually create an audit event
    
    - For administrative or special audit logging
    - Requires admin privileges
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
        
        # Create audit event
        event_id = await audit_log(
            event_type=request.event_type,
            user_id=current_user["user_id"],
            session_id=current_user.get("session_id"),
            resource=request.resource,
            action=request.action,
            outcome=request.outcome,
            details=request.details,
            data_classification=request.data_classification
        )
        
        return {
            "status": "success",
            "event_id": event_id,
            "message": "Audit event created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create audit event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/events")
async def query_audit_events(
    start_time: Optional[datetime] = Query(None, description="Start time for query"),
    end_time: Optional[datetime] = Query(None, description="End time for query"),
    event_type: Optional[AuditEventType] = Query(None, description="Filter by event type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    compliance_framework: Optional[ComplianceFramework] = Query(None, description="Filter by compliance framework"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (low, medium, high, critical)"),
    limit: int = Query(100, description="Maximum number of events to return", le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Query audit events with various filters
    
    - Supports filtering by time range, event type, user, etc.
    - Returns events in reverse chronological order
    """
    try:
        # Check permissions
        if "admin" not in current_user.get("permissions", []):
            # Non-admin users can only see their own events
            user_id = current_user["user_id"]
        
        audit_logger = get_audit_logger()
        
        # Convert single event_type to list
        event_types = [event_type] if event_type else None
        
        events = await audit_logger.query_audit_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            user_id=user_id,
            compliance_framework=compliance_framework,
            risk_level=risk_level,
            limit=limit
        )
        
        return {
            "count": len(events),
            "events": events
        }
        
    except Exception as e:
        logger.error(f"Failed to query audit events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/events/{event_id}/verify")
async def verify_audit_event(
    event_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify the integrity of a specific audit event
    
    - Checks the cryptographic checksum
    - Ensures the event hasn't been tampered with
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
        
        audit_logger = get_audit_logger()
        is_valid = await audit_logger.verify_audit_integrity(event_id)
        
        return {
            "event_id": event_id,
            "integrity_valid": is_valid,
            "verified_at": datetime.utcnow().isoformat(),
            "verified_by": current_user["user_id"]
        }
        
    except Exception as e:
        logger.error(f"Failed to verify audit event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compliance/reports")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate compliance report for specified framework
    
    - Supports SOX, GDPR, SOC2, and other frameworks
    - Analyzes audit events for compliance requirements
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
        
        # Validate date range
        if request.end_date <= request.start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        # Limit report period (max 1 year)
        max_period = timedelta(days=365)
        if request.end_date - request.start_date > max_period:
            raise HTTPException(status_code=400, detail="Report period cannot exceed 365 days")
        
        compliance_reporter = get_compliance_reporter()
        
        # Generate report based on framework
        if request.framework == ComplianceFramework.SOX:
            report = await compliance_reporter.generate_sox_report(
                request.start_date, request.end_date
            )
        elif request.framework == ComplianceFramework.GDPR:
            report = await compliance_reporter.generate_gdpr_report(
                request.start_date, request.end_date
            )
        elif request.framework == ComplianceFramework.SOC2:
            report = await compliance_reporter.generate_soc2_report(
                request.start_date, request.end_date
            )
        else:
            # Generic compliance report
            audit_logger = get_audit_logger()
            events = await audit_logger.query_audit_events(
                start_time=request.start_date,
                end_time=request.end_date,
                compliance_framework=request.framework
            )
            
            report = {
                "report_type": f"{request.framework.value}_compliance",
                "period": {
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat()
                },
                "summary": {
                    "total_events": len(events),
                    "framework": request.framework.value
                },
                "events": events
            }
        
        # Add report metadata
        report["generated_at"] = datetime.utcnow().isoformat()
        report["generated_by"] = current_user["user_id"]
        
        # Log report generation
        await audit_log(
            event_type=AuditEventType.ADMIN_ACTION,
            user_id=current_user["user_id"],
            session_id=current_user.get("session_id"),
            action="generate_compliance_report",
            details={
                "framework": request.framework.value,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "events_analyzed": report.get("summary", {}).get("total_events", 0)
            }
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/frameworks")
async def list_compliance_frameworks(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List supported compliance frameworks
    
    - Returns all available frameworks with descriptions
    """
    frameworks = {
        ComplianceFramework.SOX.value: {
            "name": "Sarbanes-Oxley Act",
            "description": "US federal law for corporate financial reporting",
            "focus_areas": ["financial_controls", "access_controls", "change_management"]
        },
        ComplianceFramework.SOC2.value: {
            "name": "SOC 2 Type II",
            "description": "Trust Service Criteria for service organizations",
            "focus_areas": ["security", "availability", "confidentiality", "processing_integrity"]
        },
        ComplianceFramework.GDPR.value: {
            "name": "General Data Protection Regulation",
            "description": "EU regulation for data protection and privacy",
            "focus_areas": ["data_processing", "consent", "data_subject_rights"]
        },
        ComplianceFramework.HIPAA.value: {
            "name": "Health Insurance Portability and Accountability Act",
            "description": "US law for healthcare data protection",
            "focus_areas": ["phi_protection", "access_controls", "audit_trails"]
        },
        ComplianceFramework.PCI_DSS.value: {
            "name": "Payment Card Industry Data Security Standard",
            "description": "Security standards for payment card data",
            "focus_areas": ["cardholder_data", "secure_networks", "access_controls"]
        },
        ComplianceFramework.ISO27001.value: {
            "name": "ISO/IEC 27001",
            "description": "International standard for information security management",
            "focus_areas": ["isms", "risk_management", "security_controls"]
        },
        ComplianceFramework.NIST.value: {
            "name": "NIST Cybersecurity Framework",
            "description": "US framework for cybersecurity risk management",
            "focus_areas": ["identify", "protect", "detect", "respond", "recover"]
        }
    }
    
    return {
        "frameworks": frameworks,
        "count": len(frameworks)
    }


@router.get("/audit/statistics")
async def get_audit_statistics(
    days: int = Query(30, description="Number of days to analyze", le=365),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get audit statistics and metrics
    
    - Event counts by type and risk level
    - Compliance framework coverage
    - User activity metrics
    """
    try:
        # Check admin permissions for full stats
        is_admin = "admin" in current_user.get("permissions", [])
        
        audit_logger = get_audit_logger()
        
        # Get events for the specified period
        start_date = datetime.utcnow() - timedelta(days=days)
        end_date = datetime.utcnow()
        
        events = await audit_logger.query_audit_events(
            start_time=start_date,
            end_time=end_date,
            user_id=None if is_admin else current_user["user_id"],
            limit=10000
        )
        
        # Calculate statistics
        event_counts = {}
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        user_activity = {}
        compliance_coverage = {}
        
        for event in events:
            # Event type counts
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Risk level counts
            risk_level = event.get("risk_level", "low")
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # User activity (admin only)
            if is_admin and event.get("user_id"):
                user_id = event["user_id"]
                user_activity[user_id] = user_activity.get(user_id, 0) + 1
            
            # Compliance framework coverage
            for framework in event.get("compliance_tags", []):
                compliance_coverage[framework] = compliance_coverage.get(framework, 0) + 1
        
        stats = {
            "period": {
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "unique_event_types": len(event_counts),
                "unique_users": len(user_activity) if is_admin else 1
            },
            "event_types": event_counts,
            "risk_distribution": risk_counts,
            "compliance_coverage": compliance_coverage
        }
        
        # Add user activity for admins only
        if is_admin:
            stats["user_activity"] = dict(sorted(
                user_activity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 most active users
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get audit statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]
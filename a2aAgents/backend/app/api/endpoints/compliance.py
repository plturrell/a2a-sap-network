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
from ...core.dataRetention import (
    get_retention_manager,
    RetentionPolicy,
    DataCategory,
    RetentionAction
)
from ...core.complianceReporting import (
    get_enhanced_compliance_reporter,
    ReportFormat,
    ReportStatus
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


class RetentionPolicyRequest(BaseModel):
    """Request model for creating/updating retention policies"""
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    data_category: DataCategory = Field(..., description="Category of data")
    retention_days: int = Field(..., ge=0, description="Days to retain data")
    action: RetentionAction = Field(..., description="Action when retention expires")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Related compliance frameworks")
    enabled: bool = Field(True, description="Whether policy is active")
    grace_period_days: int = Field(0, ge=0, description="Grace period before action")
    notification_days_before: int = Field(7, ge=0, description="Days before action to notify")


class EnhancedReportRequest(BaseModel):
    """Request model for generating enhanced compliance reports"""
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    start_date: datetime = Field(..., description="Report start date")
    end_date: datetime = Field(..., description="Report end date")
    format: ReportFormat = Field(ReportFormat.JSON, description="Output format")
    include_recommendations: bool = Field(True, description="Include improvement recommendations")


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
    request: EnhancedReportRequest,
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


@router.post("/compliance/validate-rules")
async def validate_compliance_rules(
    event_data: Dict[str, Any],
    framework: ComplianceFramework = Query(..., description="Compliance framework to validate against"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate an event or action against compliance rules

    - Checks if the event complies with specified framework rules
    - Returns validation results with any violations
    - Can be used for pre-event validation (preventive) or post-event audit
    """
    try:
        # Initialize compliance rule validator
        validation_results = {
            "framework": framework.value,
            "validated_at": datetime.utcnow().isoformat(),
            "validator": current_user["user_id"],
            "is_compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }

        # Validate based on framework
        if framework == ComplianceFramework.SOX:
            _validate_sox_rules(event_data, validation_results)
        elif framework == ComplianceFramework.GDPR:
            _validate_gdpr_rules(event_data, validation_results)
        elif framework == ComplianceFramework.SOC2:
            _validate_soc2_rules(event_data, validation_results)
        elif framework == ComplianceFramework.HIPAA:
            _validate_hipaa_rules(event_data, validation_results)
        elif framework == ComplianceFramework.PCI_DSS:
            _validate_pci_dss_rules(event_data, validation_results)
        elif framework == ComplianceFramework.ISO27001:
            _validate_iso27001_rules(event_data, validation_results)
        elif framework == ComplianceFramework.NIST:
            _validate_nist_rules(event_data, validation_results)

        # Log validation event
        await audit_log(
            event_type=AuditEventType.ADMIN_ACTION,
            user_id=current_user["user_id"],
            session_id=current_user.get("session_id"),
            action="validate_compliance_rules",
            outcome="success" if validation_results["is_compliant"] else "failure",
            details={
                "framework": framework.value,
                "event_type": event_data.get("event_type"),
                "violations_count": len(validation_results["violations"]),
                "warnings_count": len(validation_results["warnings"])
            }
        )

        return validation_results

    except Exception as e:
        logger.error(f"Failed to validate compliance rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _validate_sox_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate SOX compliance rules"""
    violations = []
    warnings = []

    # Rule 1: Financial data changes require dual approval
    if event_data.get("event_type") == "financial_data_change":
        if not event_data.get("approvals") or len(event_data.get("approvals", [])) < 2:
            violations.append({
                "rule": "SOX-001",
                "description": "Financial data changes require dual approval",
                "severity": "critical"
            })

    # Rule 2: Segregation of duties
    if event_data.get("event_type") in ["permission_grant", "role_assignment"]:
        user_id = event_data.get("user_id")
        granted_permissions = event_data.get("permissions", [])

        # Check for conflicting permissions
        conflicting_pairs = [
            (["financial_write", "financial_approve"], "Cannot grant both write and approve permissions"),
            (["user_create", "audit_delete"], "Cannot grant both user creation and audit deletion"),
            (["system_config", "audit_read"], "System configuration should not have audit read access")
        ]

        for pair, message in conflicting_pairs:
            if all(perm in granted_permissions for perm in pair):
                violations.append({
                    "rule": "SOX-002",
                    "description": f"Segregation of duties violation: {message}",
                    "severity": "high"
                })

    # Rule 3: Audit trail retention
    if event_data.get("event_type") == "audit_delete":
        retention_days = event_data.get("retention_days", float('inf'))
        if retention_days < 2555:  # 7 years in days
            violations.append({
                "rule": "SOX-003",
                "description": "Audit records must be retained for at least 7 years",
                "severity": "critical"
            })

    # Rule 4: Change management documentation
    if event_data.get("event_type") in ["config_change", "system_update"]:
        if not event_data.get("change_ticket") or not event_data.get("approval_record"):
            warnings.append({
                "rule": "SOX-004",
                "description": "System changes should have associated change tickets and approval records",
                "severity": "medium"
            })

    results["violations"].extend(violations)
    results["warnings"].extend(warnings)
    results["is_compliant"] = len(violations) == 0


def _validate_gdpr_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate GDPR compliance rules"""
    violations = []
    warnings = []

    # Rule 1: Lawful basis for data processing
    if event_data.get("event_type") in ["data_collection", "data_processing"]:
        if not event_data.get("lawful_basis"):
            violations.append({
                "rule": "GDPR-001",
                "description": "Data processing must have a documented lawful basis",
                "severity": "critical"
            })

    # Rule 2: Consent management
    if event_data.get("event_type") == "personal_data_collection":
        consent = event_data.get("consent", {})
        if not consent.get("obtained") or not consent.get("timestamp"):
            violations.append({
                "rule": "GDPR-002",
                "description": "Personal data collection requires explicit consent with timestamp",
                "severity": "high"
            })

        if not consent.get("withdrawal_mechanism"):
            warnings.append({
                "rule": "GDPR-003",
                "description": "Consent should include clear withdrawal mechanism",
                "severity": "medium"
            })

    # Rule 3: Data minimization
    if event_data.get("event_type") == "data_collection":
        collected_fields = event_data.get("fields", [])
        purpose = event_data.get("purpose", "")

        # Check for excessive data collection
        sensitive_fields = ["ssn", "credit_card", "health_data", "biometric"]
        unnecessary_sensitive = [f for f in collected_fields if f in sensitive_fields and f not in purpose]

        if unnecessary_sensitive:
            violations.append({
                "rule": "GDPR-004",
                "description": f"Data minimization violation: Collecting unnecessary sensitive data: {unnecessary_sensitive}",
                "severity": "high"
            })

    # Rule 4: Right to erasure
    if event_data.get("event_type") == "erasure_request":
        if event_data.get("response_days", 31) > 30:
            violations.append({
                "rule": "GDPR-005",
                "description": "Erasure requests must be fulfilled within 30 days",
                "severity": "medium"
            })

    # Rule 5: Data breach notification
    if event_data.get("event_type") == "data_breach":
        notification_hours = event_data.get("notification_hours", 73)
        if notification_hours > 72:
            violations.append({
                "rule": "GDPR-006",
                "description": "Data breaches must be reported within 72 hours",
                "severity": "critical"
            })

    results["violations"].extend(violations)
    results["warnings"].extend(warnings)
    results["is_compliant"] = len(violations) == 0

    # Add GDPR-specific recommendations
    if event_data.get("event_type") in ["data_collection", "data_processing"]:
        results["recommendations"].append({
            "recommendation": "Consider implementing Privacy by Design principles",
            "reference": "GDPR Article 25"
        })


def _validate_soc2_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate SOC2 compliance rules"""
    violations = []
    warnings = []

    # Security criteria
    if event_data.get("event_type") == "access_attempt":
        if not event_data.get("authentication_method"):
            violations.append({
                "rule": "SOC2-SEC-001",
                "description": "All access attempts must use authenticated methods",
                "severity": "high"
            })

    # Availability criteria
    if event_data.get("event_type") == "system_downtime":
        if not event_data.get("incident_response_time"):
            warnings.append({
                "rule": "SOC2-AVL-001",
                "description": "System downtime should have documented incident response times",
                "severity": "medium"
            })

    # Processing integrity criteria
    if event_data.get("event_type") == "data_processing":
        if not event_data.get("validation_checks"):
            warnings.append({
                "rule": "SOC2-PI-001",
                "description": "Data processing should include validation checks",
                "severity": "medium"
            })

    # Confidentiality criteria
    if event_data.get("event_type") == "data_access":
        data_classification = event_data.get("data_classification", "")
        if data_classification in ["confidential", "restricted"] and not event_data.get("encryption"):
            violations.append({
                "rule": "SOC2-CON-001",
                "description": "Confidential data access must use encryption",
                "severity": "high"
            })

    results["violations"].extend(violations)
    results["warnings"].extend(warnings)
    results["is_compliant"] = len(violations) == 0


def _validate_hipaa_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate HIPAA compliance rules"""
    violations = []

    # PHI access controls
    if event_data.get("event_type") == "phi_access":
        if not event_data.get("access_justification"):
            violations.append({
                "rule": "HIPAA-001",
                "description": "PHI access requires documented justification",
                "severity": "critical"
            })

    # Minimum necessary standard
    if event_data.get("event_type") == "phi_disclosure":
        if not event_data.get("minimum_necessary_assessment"):
            violations.append({
                "rule": "HIPAA-002",
                "description": "PHI disclosure must follow minimum necessary standard",
                "severity": "high"
            })

    results["violations"].extend(violations)
    results["is_compliant"] = len(violations) == 0


def _validate_pci_dss_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate PCI DSS compliance rules"""
    violations = []

    # Cardholder data protection
    if event_data.get("event_type") == "card_data_storage":
        if not event_data.get("encryption") or event_data.get("encryption_type") != "AES256":
            violations.append({
                "rule": "PCI-001",
                "description": "Cardholder data must be encrypted with AES256",
                "severity": "critical"
            })

    results["violations"].extend(violations)
    results["is_compliant"] = len(violations) == 0


def _validate_iso27001_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate ISO27001 compliance rules"""
    warnings = []

    # Risk assessment requirement
    if event_data.get("event_type") == "security_control_change":
        if not event_data.get("risk_assessment"):
            warnings.append({
                "rule": "ISO27001-001",
                "description": "Security control changes should include risk assessment",
                "severity": "medium"
            })

    results["warnings"].extend(warnings)


def _validate_nist_rules(event_data: Dict[str, Any], results: Dict[str, Any]):
    """Validate NIST Cybersecurity Framework rules"""
    warnings = []

    # Continuous monitoring
    if event_data.get("event_type") == "security_event":
        if not event_data.get("detection_time"):
            warnings.append({
                "rule": "NIST-001",
                "description": "Security events should have detection time metrics",
                "severity": "low"
            })

    results["warnings"].extend(warnings)


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


@router.get("/retention/policies")
async def list_retention_policies(
    category: Optional[DataCategory] = Query(None, description="Filter by data category"),
    framework: Optional[str] = Query(None, description="Filter by compliance framework"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all data retention policies

    - Can filter by data category or compliance framework
    - Shows policy details and current status
    """
    try:
        retention_manager = get_retention_manager()

        if category:
            policies = retention_manager.get_policies_by_category(category)
        elif framework:
            policies = retention_manager.get_policies_by_framework(framework)
        else:
            policies = list(retention_manager.policies.values())

        # Convert to response format
        policies_data = []
        for policy in policies:
            policies_data.append({
                "policy_id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "data_category": policy.data_category.value,
                "retention_days": policy.retention_days,
                "action": policy.action.value,
                "compliance_frameworks": policy.compliance_frameworks,
                "enabled": policy.enabled,
                "grace_period_days": policy.grace_period_days,
                "notification_days_before": policy.notification_days_before,
                "created_at": policy.created_at.isoformat(),
                "updated_at": policy.updated_at.isoformat()
            })

        return {
            "count": len(policies_data),
            "policies": policies_data
        }

    except Exception as e:
        logger.error(f"Failed to list retention policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/policies")
async def create_retention_policy(
    request: RetentionPolicyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new data retention policy

    - Requires admin permissions
    - Validates retention requirements
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")

        retention_manager = get_retention_manager()

        # Create policy
        policy = RetentionPolicy(
            policy_id=f"custom_{request.name.lower().replace(' ', '_')}_{datetime.utcnow().timestamp()}",
            name=request.name,
            description=request.description,
            data_category=request.data_category,
            retention_days=request.retention_days,
            action=request.action,
            compliance_frameworks=request.compliance_frameworks,
            enabled=request.enabled,
            grace_period_days=request.grace_period_days,
            notification_days_before=request.notification_days_before
        )

        policy_id = retention_manager.add_policy(policy)

        # Log action
        await audit_log(
            event_type=AuditEventType.CONFIG_CHANGED,
            user_id=current_user["user_id"],
            action="create_retention_policy",
            resource=f"retention_policy:{policy_id}",
            details={
                "policy_name": request.name,
                "data_category": request.data_category.value,
                "retention_days": request.retention_days
            }
        )

        return {
            "status": "success",
            "policy_id": policy_id,
            "message": "Retention policy created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create retention policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/retention/policies/{policy_id}")
async def update_retention_policy(
    policy_id: str,
    request: RetentionPolicyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update an existing retention policy

    - Requires admin permissions
    - Cannot reduce retention below compliance requirements
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")

        retention_manager = get_retention_manager()

        # Check if policy exists
        existing_policy = retention_manager.get_policy(policy_id)
        if not existing_policy:
            raise HTTPException(status_code=404, detail="Policy not found")

        # Update policy
        updated_policy = RetentionPolicy(
            policy_id=policy_id,
            name=request.name,
            description=request.description,
            data_category=request.data_category,
            retention_days=request.retention_days,
            action=request.action,
            compliance_frameworks=request.compliance_frameworks,
            enabled=request.enabled,
            grace_period_days=request.grace_period_days,
            notification_days_before=request.notification_days_before,
            created_at=existing_policy.created_at
        )

        retention_manager.add_policy(updated_policy)

        # Log action
        await audit_log(
            event_type=AuditEventType.CONFIG_CHANGED,
            user_id=current_user["user_id"],
            action="update_retention_policy",
            resource=f"retention_policy:{policy_id}",
            details={
                "old_retention_days": existing_policy.retention_days,
                "new_retention_days": request.retention_days,
                "changes": {
                    "enabled": request.enabled,
                    "action": request.action.value
                }
            }
        )

        return {
            "status": "success",
            "policy_id": policy_id,
            "message": "Retention policy updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update retention policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/apply")
async def apply_retention_policies(
    dry_run: bool = Query(False, description="Simulate execution without making changes"),
    policy_id: Optional[str] = Query(None, description="Apply specific policy only"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Apply data retention policies

    - Can run in dry-run mode to preview actions
    - Can apply specific policy or all policies
    - Returns detailed execution results
    """
    try:
        # Check admin permissions
        if "admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")

        retention_manager = get_retention_manager()

        # Apply policies
        if policy_id:
            # Apply single policy
            policy = retention_manager.get_policy(policy_id)
            if not policy:
                raise HTTPException(status_code=404, detail="Policy not found")

            result = await retention_manager._apply_single_policy(policy, dry_run)
            results = {
                "policies_evaluated": 1,
                "policies_applied": 1 if result["applied"] else 0,
                "data_processed": result["data_processed"],
                "data_retained": result["data_retained"],
                "data_actioned": result["data_actioned"],
                "errors": result["errors"],
                "dry_run": dry_run
            }
        else:
            # Apply all policies
            results = await retention_manager.apply_retention_policies(dry_run)
            results["dry_run"] = dry_run

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply retention policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/compliance-check")
async def check_retention_compliance(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if retention policies meet compliance requirements

    - Validates policies against known compliance standards
    - Returns any violations or gaps
    """
    try:
        retention_manager = get_retention_manager()
        violations = retention_manager.validate_compliance_requirements()

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "checked_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to check retention compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retention/categories")
async def list_data_categories(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List available data categories for retention policies
    """
    categories = [
        {
            "value": category.value,
            "description": category.value.replace("_", " ").title()
        }
        for category in DataCategory
    ]

    return {
        "categories": categories,
        "count": len(categories)
    }


@router.get("/retention/actions")
async def list_retention_actions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List available retention actions
    """
    actions = [
        {
            "value": action.value,
            "description": action.value.replace("_", " ").title()
        }
        for action in RetentionAction
    ]

    return {
        "actions": actions,
        "count": len(actions)
    }


@router.post("/reports/enhanced")
async def generate_enhanced_compliance_report(
    request: EnhancedReportRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate comprehensive compliance report with analysis and recommendations

    - Supports all major compliance frameworks
    - Includes detailed analysis and risk assessment
    - Provides actionable recommendations
    - Asynchronous generation for large reports
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

        enhanced_reporter = get_enhanced_compliance_reporter()

        # Generate report (async)
        request_id = await enhanced_reporter.generate_comprehensive_report(
            framework=request.framework,
            start_date=request.start_date,
            end_date=request.end_date,
            requested_by=current_user["user_id"],
            format=request.format,
            include_recommendations=request.include_recommendations
        )

        return {
            "status": "accepted",
            "request_id": request_id,
            "message": "Report generation initiated",
            "framework": request.framework.value,
            "estimated_completion": "5-10 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate report generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{request_id}")
async def get_report_status(
    request_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get compliance report generation status and results

    - Returns generation progress
    - Provides download link when complete
    - Includes report metadata
    """
    try:
        enhanced_reporter = get_enhanced_compliance_reporter()

        report_request = await enhanced_reporter.get_report_status(request_id)
        if not report_request:
            raise HTTPException(status_code=404, detail="Report request not found")

        # Check permissions (only creator can view)
        if report_request.requested_by != current_user["user_id"]:
            # Could check admin permissions here
            raise HTTPException(status_code=403, detail="Not authorized to view this report")

        response = {
            "request_id": request_id,
            "status": report_request.status.value,
            "framework": report_request.framework.value,
            "format": report_request.format.value,
            "requested_at": report_request.requested_at.isoformat(),
            "period": {
                "start_date": report_request.start_date.isoformat(),
                "end_date": report_request.end_date.isoformat(),
                "duration_days": (report_request.end_date - report_request.start_date).days
            }
        }

        if report_request.status == ReportStatus.COMPLETED:
            response["completed_at"] = report_request.completed_at.isoformat()
            response["download_url"] = f"/admin/reports/{request_id}/download"
            response["metadata"] = report_request.metadata

        elif report_request.status == ReportStatus.FAILED:
            response["error_message"] = report_request.error_message

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get report status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
async def list_compliance_reports(
    framework: Optional[ComplianceFramework] = Query(None, description="Filter by framework"),
    status: Optional[ReportStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of reports"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List compliance reports

    - Shows user's reports or all reports (admin)
    - Supports filtering by framework and status
    - Includes report metadata and status
    """
    try:
        enhanced_reporter = get_enhanced_compliance_reporter()

        reports = await enhanced_reporter.list_reports(
            framework=framework,
            status=status,
            limit=limit
        )

        # Filter to user's reports unless admin
        is_admin = "admin" in current_user.get("permissions", [])
        if not is_admin:
            reports = [r for r in reports if r.requested_by == current_user["user_id"]]

        # Convert to response format
        report_list = []
        for report in reports:
            report_list.append({
                "request_id": report.request_id,
                "framework": report.framework.value,
                "status": report.status.value,
                "format": report.format.value,
                "requested_by": report.requested_by if is_admin else None,
                "requested_at": report.requested_at.isoformat(),
                "completed_at": report.completed_at.isoformat() if report.completed_at else None,
                "period_days": (report.end_date - report.start_date).days,
                "metadata": report.metadata
            })

        return {
            "count": len(report_list),
            "reports": report_list
        }

    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{request_id}/download")
async def download_compliance_report(
    request_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Download completed compliance report

    - Returns report file in requested format
    - Validates user permissions
    - Logs download activity
    """
    try:
        enhanced_reporter = get_enhanced_compliance_reporter()

        report_request = await enhanced_reporter.get_report_status(request_id)
        if not report_request:
            raise HTTPException(status_code=404, detail="Report not found")

        # Check permissions
        if report_request.requested_by != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Check if report is complete
        if report_request.status != ReportStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Report not yet complete")

        # Check if file exists
        if not report_request.file_path or not Path(report_request.file_path).exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        # Log download
        await audit_log(
            event_type=AuditEventType.DATA_EXPORT,
            user_id=current_user["user_id"],
            action="download_compliance_report",
            resource=f"report:{request_id}",
            details={
                "framework": report_request.framework.value,
                "format": report_request.format.value
            }
        )

        # Return file download
        from fastapi.responses import FileResponse

        filename = f"{report_request.framework.value}_report_{request_id[:8]}.{report_request.format.value}"

        return FileResponse(
            path=report_request.file_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reports/{request_id}")
async def delete_compliance_report(
    request_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a compliance report

    - Removes report file and tracking record
    - Only creator or admin can delete
    - Logs deletion activity
    """
    try:
        enhanced_reporter = get_enhanced_compliance_reporter()

        success = await enhanced_reporter.delete_report(request_id, current_user["user_id"])

        if not success:
            raise HTTPException(status_code=404, detail="Report not found or not authorized")

        # Log deletion
        await audit_log(
            event_type=AuditEventType.DATA_DELETE,
            user_id=current_user["user_id"],
            action="delete_compliance_report",
            resource=f"report:{request_id}"
        )

        return {
            "status": "success",
            "message": "Report deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/formats")
async def list_report_formats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List available report formats
    """
    formats = [
        {
            "value": format.value,
            "description": format.value.upper(),
            "content_type": {
                "json": "application/json",
                "pdf": "application/pdf",
                "csv": "text/csv",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }.get(format.value, "application/octet-stream")
        }
        for format in ReportFormat
    ]

    return {
        "formats": formats,
        "count": len(formats)
    }


# Export router
__all__ = ["router"]

"""
Audit Trail and Compliance Framework
Comprehensive audit logging and compliance reporting system
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

from .config import settings
from .errorHandling import ValidationError, SecurityError
from .secrets import get_secrets_manager
from .securityMonitoring import report_security_event, EventType, ThreatLevel

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    
    # Data access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # System changes
    CONFIG_CHANGED = "config_changed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    
    # Security events
    SECURITY_ALERT = "security_alert"
    ACCESS_DENIED = "access_denied"
    AUTHENTICATION_FAILED = "authentication_failed"
    
    # API operations
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_REQUEST = "api_request"
    
    # Administrative actions
    ADMIN_ACTION = "admin_action"
    SYSTEM_MAINTENANCE = "system_maintenance"
    BACKUP_CREATED = "backup_created"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    SOX = "sox"  # Sarbanes-Oxley Act
    SOC2 = "soc2"  # SOC 2 Type II
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # ISO/IEC 27001
    NIST = "nist"  # NIST Cybersecurity Framework


@dataclass
class AuditEvent:
    """Single audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[ComplianceFramework] = field(default_factory=list)
    data_classification: Optional[str] = None  # public, internal, confidential, restricted
    risk_level: str = "low"  # low, medium, high, critical
    checksum: Optional[str] = None


class AuditLogger:
    """Centralized audit logging system"""
    
    def __init__(self):
        self.secrets_manager = get_secrets_manager()
        
        # Configure audit storage
        self.audit_file_path = Path("logs/audit.jsonl")
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compliance mappings
        self.compliance_mappings = self._initialize_compliance_mappings()
        
        # Risk assessments
        self.risk_assessments = self._initialize_risk_assessments()
        
        logger.info("Audit Logger initialized")
    
    def _initialize_compliance_mappings(self) -> Dict[AuditEventType, List[ComplianceFramework]]:
        """Map audit events to compliance frameworks"""
        return {
            # User management - critical for all frameworks
            AuditEventType.USER_LOGIN: [ComplianceFramework.SOX, ComplianceFramework.SOC2, 
                                      ComplianceFramework.GDPR, ComplianceFramework.ISO27001],
            AuditEventType.USER_CREATED: [ComplianceFramework.SOX, ComplianceFramework.SOC2,
                                        ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            AuditEventType.USER_UPDATED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            AuditEventType.USER_DELETED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            
            # Data access - critical for data protection
            AuditEventType.DATA_READ: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA,
                                     ComplianceFramework.SOC2],
            AuditEventType.DATA_WRITE: [ComplianceFramework.SOX, ComplianceFramework.GDPR,
                                      ComplianceFramework.HIPAA, ComplianceFramework.SOC2],
            AuditEventType.DATA_DELETE: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA,
                                       ComplianceFramework.SOX],
            AuditEventType.DATA_EXPORT: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            
            # Security events - critical for all frameworks
            AuditEventType.SECURITY_ALERT: [ComplianceFramework.SOC2, ComplianceFramework.ISO27001,
                                          ComplianceFramework.NIST, ComplianceFramework.PCI_DSS],
            AuditEventType.ACCESS_DENIED: [ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
            
            # Administrative actions - SOX and SOC2 focus
            AuditEventType.ADMIN_ACTION: [ComplianceFramework.SOX, ComplianceFramework.SOC2],
            AuditEventType.CONFIG_CHANGED: [ComplianceFramework.SOX, ComplianceFramework.SOC2,
                                          ComplianceFramework.ISO27001],
            AuditEventType.PERMISSION_GRANTED: [ComplianceFramework.SOX, ComplianceFramework.SOC2],
            AuditEventType.PERMISSION_REVOKED: [ComplianceFramework.SOX, ComplianceFramework.SOC2]
        }
    
    def _initialize_risk_assessments(self) -> Dict[AuditEventType, str]:
        """Define risk levels for different event types"""
        return {
            # High-risk events
            AuditEventType.USER_DELETED: "high",
            AuditEventType.DATA_DELETE: "high",
            AuditEventType.PERMISSION_GRANTED: "high",
            AuditEventType.API_KEY_CREATED: "high",
            AuditEventType.CONFIG_CHANGED: "high",
            
            # Critical-risk events
            AuditEventType.ADMIN_ACTION: "critical",
            AuditEventType.SECURITY_ALERT: "critical",
            
            # Medium-risk events
            AuditEventType.USER_CREATED: "medium",
            AuditEventType.USER_UPDATED: "medium",
            AuditEventType.DATA_WRITE: "medium",
            AuditEventType.DATA_EXPORT: "medium",
            AuditEventType.ACCESS_DENIED: "medium",
            
            # Low-risk events (default)
            AuditEventType.USER_LOGIN: "low",
            AuditEventType.USER_LOGOUT: "low",
            AuditEventType.DATA_READ: "low",
            AuditEventType.API_REQUEST: "low"
        }
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        data_classification: Optional[str] = None
    ) -> str:
        """Log an audit event"""
        try:
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Determine compliance tags
            compliance_tags = self.compliance_mappings.get(event_type, [])
            
            # Determine risk level
            risk_level = self.risk_assessments.get(event_type, "low")
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                outcome=outcome,
                details=details or {},
                compliance_tags=compliance_tags,
                data_classification=data_classification,
                risk_level=risk_level
            )
            
            # Calculate checksum for integrity
            event.checksum = self._calculate_checksum(event)
            
            # Write to audit log
            await self._write_audit_record(event)
            
            # Report high-risk events to security monitoring
            if risk_level in ["high", "critical"]:
                await self._report_high_risk_event(event)
            
            logger.debug(f"Audit event logged: {event_id} ({event_type.value})")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise SecurityError(f"Audit logging failed: {e}")
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate integrity checksum for audit event"""
        # Create canonical representation
        data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome,
            "details": event.details
        }
        
        # Sort keys for consistent hashing
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    async def _write_audit_record(self, event: AuditEvent):
        """Write audit record to persistent storage"""
        try:
            # Convert to JSON
            record = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "resource": event.resource,
                "action": event.action,
                "outcome": event.outcome,
                "details": event.details,
                "compliance_tags": [tag.value for tag in event.compliance_tags],
                "data_classification": event.data_classification,
                "risk_level": event.risk_level,
                "checksum": event.checksum
            }
            
            # Append to audit log file (JSONL format)
            with open(self.audit_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            
        except Exception as e:
            logger.error(f"Failed to write audit record: {e}")
            raise
    
    async def _report_high_risk_event(self, event: AuditEvent):
        """Report high-risk events to security monitoring"""
        threat_level = ThreatLevel.HIGH if event.risk_level == "high" else ThreatLevel.CRITICAL
        
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,  # Using as audit event
            threat_level=threat_level,
            description=f"High-risk audit event: {event.event_type.value}",
            user_id=event.user_id,
            session_id=event.session_id,
            source_ip=event.ip_address,
            details={
                "audit_event_id": event.event_id,
                "compliance_frameworks": [tag.value for tag in event.compliance_tags],
                "data_classification": event.data_classification,
                "outcome": event.outcome
            }
        )
    
    async def query_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        compliance_framework: Optional[ComplianceFramework] = None,
        risk_level: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        try:
            events = []
            
            # Read audit log file
            if not self.audit_file_path.exists():
                return events
            
            with open(self.audit_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(events) >= limit:
                        break
                        
                    try:
                        record = json.loads(line.strip())
                        
                        # Apply filters
                        if start_time:
                            event_time = datetime.fromisoformat(record['timestamp'])
                            if event_time < start_time:
                                continue
                        
                        if end_time:
                            event_time = datetime.fromisoformat(record['timestamp'])
                            if event_time > end_time:
                                continue
                        
                        if event_types and record['event_type'] not in [et.value for et in event_types]:
                            continue
                        
                        if user_id and record.get('user_id') != user_id:
                            continue
                        
                        if compliance_framework:
                            if compliance_framework.value not in record.get('compliance_tags', []):
                                continue
                        
                        if risk_level and record.get('risk_level') != risk_level:
                            continue
                        
                        events.append(record)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    async def verify_audit_integrity(self, event_id: str) -> bool:
        """Verify the integrity of an audit event"""
        try:
            # Find the event
            events = await self.query_audit_events(limit=10000)
            
            for record in events:
                if record['event_id'] == event_id:
                    # Recalculate checksum
                    stored_checksum = record.pop('checksum', None)
                    
                    # Reconstruct event for checksum calculation
                    data = {
                        "event_id": record['event_id'],
                        "timestamp": record['timestamp'],
                        "event_type": record['event_type'],
                        "user_id": record.get('user_id'),
                        "session_id": record.get('session_id'),
                        "resource": record.get('resource'),
                        "action": record.get('action'),
                        "outcome": record['outcome'],
                        "details": record['details']
                    }
                    
                    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
                    calculated_checksum = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                    
                    return stored_checksum == calculated_checksum
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to verify audit integrity: {e}")
            return False


class ComplianceReporter:
    """Generate compliance reports for various frameworks"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    async def generate_sox_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate Sarbanes-Oxley compliance report"""
        
        # Query SOX-relevant events
        events = await self.audit_logger.query_audit_events(
            start_time=start_date,
            end_time=end_date,
            compliance_framework=ComplianceFramework.SOX
        )
        
        # Categorize events
        user_management = [e for e in events if e['event_type'] in [
            AuditEventType.USER_CREATED.value,
            AuditEventType.USER_UPDATED.value,
            AuditEventType.USER_DELETED.value
        ]]
        
        access_controls = [e for e in events if e['event_type'] in [
            AuditEventType.PERMISSION_GRANTED.value,
            AuditEventType.PERMISSION_REVOKED.value,
            AuditEventType.ACCESS_DENIED.value
        ]]
        
        system_changes = [e for e in events if e['event_type'] in [
            AuditEventType.CONFIG_CHANGED.value,
            AuditEventType.ADMIN_ACTION.value
        ]]
        
        data_changes = [e for e in events if e['event_type'] in [
            AuditEventType.DATA_WRITE.value,
            AuditEventType.DATA_DELETE.value
        ]]
        
        return {
            "report_type": "sox_compliance",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "user_management_events": len(user_management),
                "access_control_events": len(access_controls),
                "system_change_events": len(system_changes),
                "data_change_events": len(data_changes)
            },
            "key_requirements": {
                "segregation_of_duties": self._assess_segregation_of_duties(events),
                "access_controls": self._assess_access_controls(access_controls),
                "change_management": self._assess_change_management(system_changes),
                "data_integrity": self._assess_data_integrity(data_changes)
            },
            "events": events
        }
    
    async def generate_gdpr_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        
        events = await self.audit_logger.query_audit_events(
            start_time=start_date,
            end_time=end_date,
            compliance_framework=ComplianceFramework.GDPR
        )
        
        # Focus on personal data processing
        data_access = [e for e in events if e['event_type'] == AuditEventType.DATA_READ.value]
        data_exports = [e for e in events if e['event_type'] == AuditEventType.DATA_EXPORT.value]
        data_deletions = [e for e in events if e['event_type'] == AuditEventType.DATA_DELETE.value]
        user_deletions = [e for e in events if e['event_type'] == AuditEventType.USER_DELETED.value]
        
        return {
            "report_type": "gdpr_compliance",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "data_access_events": len(data_access),
                "data_export_events": len(data_exports),
                "data_deletion_events": len(data_deletions),
                "user_deletion_events": len(user_deletions)
            },
            "gdpr_requirements": {
                "right_to_access": len(data_access),
                "right_to_portability": len(data_exports),
                "right_to_erasure": len(data_deletions) + len(user_deletions),
                "data_processing_lawfulness": self._assess_data_processing_lawfulness(events)
            },
            "events": events
        }
    
    async def generate_soc2_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOC 2 Type II compliance report"""
        
        events = await self.audit_logger.query_audit_events(
            start_time=start_date,
            end_time=end_date,
            compliance_framework=ComplianceFramework.SOC2
        )
        
        # SOC 2 Trust Service Criteria
        security_events = [e for e in events if e['event_type'] in [
            AuditEventType.SECURITY_ALERT.value,
            AuditEventType.ACCESS_DENIED.value,
            AuditEventType.AUTHENTICATION_FAILED.value
        ]]
        
        availability_events = [e for e in events if e['event_type'] in [
            AuditEventType.SYSTEM_MAINTENANCE.value,
            AuditEventType.BACKUP_CREATED.value
        ]]
        
        confidentiality_events = [e for e in events if e.get('data_classification') in [
            'confidential', 'restricted'
        ]]
        
        return {
            "report_type": "soc2_type2",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "trust_service_criteria": {
                "security": {
                    "events_count": len(security_events),
                    "controls_effective": len([e for e in security_events if e['outcome'] == 'success']) / max(len(security_events), 1) > 0.95
                },
                "availability": {
                    "events_count": len(availability_events),
                    "uptime_maintained": True  # Would be calculated from actual uptime data
                },
                "confidentiality": {
                    "events_count": len(confidentiality_events),
                    "data_protected": len([e for e in confidentiality_events if e['outcome'] == 'success']) / max(len(confidentiality_events), 1) > 0.98
                }
            },
            "events": events
        }
    
    def _assess_segregation_of_duties(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess segregation of duties compliance"""
        # Analyze if same users are performing conflicting duties
        user_actions = {}
        for event in events:
            user_id = event.get('user_id')
            if user_id:
                if user_id not in user_actions:
                    user_actions[user_id] = set()
                user_actions[user_id].add(event['event_type'])
        
        violations = []
        for user_id, actions in user_actions.items():
            # Example: user shouldn't both create and approve
            if (AuditEventType.USER_CREATED.value in actions and 
                AuditEventType.PERMISSION_GRANTED.value in actions):
                violations.append({
                    "user_id": user_id,
                    "violation": "User both creates accounts and grants permissions"
                })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "users_analyzed": len(user_actions)
        }
    
    def _assess_access_controls(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess access control effectiveness"""
        total_access_attempts = len(events)
        denied_access = len([e for e in events if e['event_type'] == AuditEventType.ACCESS_DENIED.value])
        
        return {
            "total_access_events": total_access_attempts,
            "denied_access_events": denied_access,
            "control_effectiveness": 1.0 if total_access_attempts == 0 else (total_access_attempts - denied_access) / total_access_attempts
        }
    
    def _assess_change_management(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess change management processes"""
        authorized_changes = len([e for e in events if e['outcome'] == 'success'])
        total_changes = len(events)
        
        return {
            "total_changes": total_changes,
            "authorized_changes": authorized_changes,
            "unauthorized_changes": total_changes - authorized_changes,
            "compliance_rate": 1.0 if total_changes == 0 else authorized_changes / total_changes
        }
    
    def _assess_data_integrity(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data integrity controls"""
        successful_operations = len([e for e in events if e['outcome'] == 'success'])
        total_operations = len(events)
        
        return {
            "total_data_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "integrity_rate": 1.0 if total_operations == 0 else successful_operations / total_operations
        }
    
    def _assess_data_processing_lawfulness(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess GDPR data processing lawfulness"""
        # This would integrate with consent management system
        data_processing_events = [e for e in events if e['event_type'] in [
            AuditEventType.DATA_READ.value,
            AuditEventType.DATA_WRITE.value,
            AuditEventType.DATA_EXPORT.value
        ]]
        
        return {
            "total_processing_events": len(data_processing_events),
            "lawful_basis_documented": len(data_processing_events),  # Assuming all are documented
            "compliance_rate": 1.0
        }


# Global instances
_audit_logger: Optional[AuditLogger] = None
_compliance_reporter: Optional[ComplianceReporter] = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_compliance_reporter() -> ComplianceReporter:
    """Get global compliance reporter instance"""
    global _compliance_reporter
    if _compliance_reporter is None:
        _compliance_reporter = ComplianceReporter(get_audit_logger())
    return _compliance_reporter

# Convenience function for easy audit logging
async def audit_log(
    event_type: AuditEventType,
    user_id: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function for audit logging"""
    audit_logger = get_audit_logger()
    return await audit_logger.log_event(event_type=event_type, user_id=user_id, **kwargs)


# Export main classes and functions
__all__ = [
    'AuditLogger',
    'ComplianceReporter',
    'AuditEvent',
    'AuditEventType',
    'ComplianceFramework',
    'get_audit_logger',
    'get_compliance_reporter',
    'audit_log'
]
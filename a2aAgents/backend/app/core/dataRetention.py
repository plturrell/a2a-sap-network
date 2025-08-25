"""
Data Retention Policy Management
Implements configurable data retention policies for compliance
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import time

from .config import settings
from .auditTrail import get_audit_logger, audit_log, AuditEventType
from .errorHandling import ValidationError
from .securityMonitoring import report_security_event, EventType, ThreatLevel
import hashlib

logger = logging.getLogger(__name__)


class DataCategory(str, Enum):
    """Categories of data with different retention requirements"""
    AUDIT_LOGS = "audit_logs"
    USER_DATA = "user_data"
    SESSION_DATA = "session_data"
    SECURITY_EVENTS = "security_events"
    API_LOGS = "api_logs"
    FINANCIAL_RECORDS = "financial_records"
    COMPLIANCE_REPORTS = "compliance_reports"
    TEMPORARY_DATA = "temporary_data"
    BACKUP_DATA = "backup_data"
    ANALYTICS_DATA = "analytics_data"


class RetentionAction(str, Enum):
    """Actions to take when retention period expires"""
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    COMPRESS = "compress"
    MOVE_COLD_STORAGE = "move_cold_storage"


@dataclass
class RetentionPolicy:
    """Defines a data retention policy"""
    policy_id: str
    name: str
    description: str
    data_category: DataCategory
    retention_days: int
    action: RetentionAction
    compliance_frameworks: List[str] = field(default_factory=list)
    enabled: bool = True
    grace_period_days: int = 0
    notification_days_before: int = 7
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class DataRetentionManager:
    """Manages data retention policies and enforcement"""

    def __init__(self):
        self.policies: Dict[str, RetentionPolicy] = {}
        self.policy_file = Path("config/retention_policies.json")
        self.execution_log = Path("logs/retention_execution.jsonl")

        # Initialize default policies
        self._initialize_default_policies()

        # Load custom policies
        self._load_policies()

        logger.info("Data Retention Manager initialized")

    def _initialize_default_policies(self):
        """Initialize default retention policies based on compliance requirements"""
        default_policies = [
            # SOX Requirements
            RetentionPolicy(
                policy_id="sox_audit_retention",
                name="SOX Audit Log Retention",
                description="Retain audit logs for 7 years per SOX requirements",
                data_category=DataCategory.AUDIT_LOGS,
                retention_days=2555,  # 7 years
                action=RetentionAction.ARCHIVE,
                compliance_frameworks=["SOX"],
                notification_days_before=30
            ),

            # GDPR Requirements
            RetentionPolicy(
                policy_id="gdpr_user_data",
                name="GDPR User Data Retention",
                description="Delete personal data when no longer needed",
                data_category=DataCategory.USER_DATA,
                retention_days=1095,  # 3 years
                action=RetentionAction.ANONYMIZE,
                compliance_frameworks=["GDPR"],
                grace_period_days=30
            ),

            # Security Best Practices
            RetentionPolicy(
                policy_id="session_cleanup",
                name="Session Data Cleanup",
                description="Remove expired session data",
                data_category=DataCategory.SESSION_DATA,
                retention_days=30,
                action=RetentionAction.DELETE,
                compliance_frameworks=["SOC2", "ISO27001"]
            ),

            RetentionPolicy(
                policy_id="security_event_retention",
                name="Security Event Retention",
                description="Retain security events for investigation",
                data_category=DataCategory.SECURITY_EVENTS,
                retention_days=365,
                action=RetentionAction.COMPRESS,
                compliance_frameworks=["SOC2", "ISO27001", "NIST"]
            ),

            # Financial Records
            RetentionPolicy(
                policy_id="financial_records",
                name="Financial Records Retention",
                description="Retain financial records per regulatory requirements",
                data_category=DataCategory.FINANCIAL_RECORDS,
                retention_days=3650,  # 10 years
                action=RetentionAction.MOVE_COLD_STORAGE,
                compliance_frameworks=["SOX", "PCI-DSS"]
            ),

            # Temporary Data
            RetentionPolicy(
                policy_id="temp_data_cleanup",
                name="Temporary Data Cleanup",
                description="Clean up temporary processing data",
                data_category=DataCategory.TEMPORARY_DATA,
                retention_days=7,
                action=RetentionAction.DELETE,
                compliance_frameworks=[]
            )
        ]

        for policy in default_policies:
            self.policies[policy.policy_id] = policy

    def _load_policies(self):
        """Load custom policies from configuration file"""
        if self.policy_file.exists():
            try:
                with open(self.policy_file, 'r') as f:
                    custom_policies = json.load(f)

                for policy_data in custom_policies:
                    policy = RetentionPolicy(**policy_data)
                    self.policies[policy.policy_id] = policy

                logger.info(f"Loaded {len(custom_policies)} custom retention policies")

            except Exception as e:
                logger.error(f"Failed to load custom policies: {e}")

    def add_policy(self, policy: RetentionPolicy) -> str:
        """Add or update a retention policy"""
        if policy.retention_days < 0:
            raise ValidationError("Retention days must be non-negative")

        if policy.policy_id in self.policies:
            logger.info(f"Updating existing policy: {policy.policy_id}")
        else:
            logger.info(f"Adding new policy: {policy.policy_id}")

        policy.updated_at = datetime.utcnow()
        self.policies[policy.policy_id] = policy

        # Save policies
        self._save_policies()

        return policy.policy_id

    def _save_policies(self):
        """Save current policies to file"""
        try:
            self.policy_file.parent.mkdir(parents=True, exist_ok=True)

            policies_data = []
            for policy in self.policies.values():
                policy_dict = {
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
                }
                policies_data.append(policy_dict)

            with open(self.policy_file, 'w') as f:
                json.dump(policies_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save policies: {e}")

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a specific retention policy"""
        return self.policies.get(policy_id)

    def get_policies_by_category(self, category: DataCategory) -> List[RetentionPolicy]:
        """Get all policies for a specific data category"""
        return [
            policy for policy in self.policies.values()
            if policy.data_category == category and policy.enabled
        ]

    def get_policies_by_framework(self, framework: str) -> List[RetentionPolicy]:
        """Get all policies for a specific compliance framework"""
        return [
            policy for policy in self.policies.values()
            if framework in policy.compliance_frameworks and policy.enabled
        ]

    async def apply_retention_policies(self, dry_run: bool = False) -> Dict[str, Any]:
        """Apply all active retention policies"""
        results = {
            "start_time": datetime.utcnow().isoformat(),
            "policies_evaluated": 0,
            "policies_applied": 0,
            "data_processed": 0,
            "data_retained": 0,
            "data_actioned": 0,
            "errors": [],
            "actions": []
        }

        try:
            for policy in self.policies.values():
                if not policy.enabled:
                    continue

                results["policies_evaluated"] += 1

                # Apply the policy
                policy_result = await self._apply_single_policy(policy, dry_run)

                if policy_result["applied"]:
                    results["policies_applied"] += 1

                results["data_processed"] += policy_result["data_processed"]
                results["data_retained"] += policy_result["data_retained"]
                results["data_actioned"] += policy_result["data_actioned"]

                if policy_result["errors"]:
                    results["errors"].extend(policy_result["errors"])

                results["actions"].append({
                    "policy_id": policy.policy_id,
                    "result": policy_result
                })

            results["end_time"] = datetime.utcnow().isoformat()

            # Log execution
            await self._log_execution(results)

            # Audit event
            await audit_log(
                event_type=AuditEventType.ADMIN_ACTION,
                action="apply_retention_policies",
                outcome="success" if not results["errors"] else "partial",
                details=results
            )

            return results

        except Exception as e:
            logger.error(f"Failed to apply retention policies: {e}")
            results["errors"].append(str(e))
            return results

    async def _apply_single_policy(
        self,
        policy: RetentionPolicy,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Apply a single retention policy"""
        result = {
            "applied": False,
            "data_processed": 0,
            "data_retained": 0,
            "data_actioned": 0,
            "errors": []
        }

        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)

            # Get data items for this category
            data_items = await self._get_data_items(policy.data_category, cutoff_date)

            result["data_processed"] = len(data_items)

            # Check for items within grace period
            grace_cutoff = cutoff_date + timedelta(days=policy.grace_period_days)

            for item in data_items:
                item_date = item.get("created_at", item.get("timestamp"))

                if isinstance(item_date, str):
                    item_date = datetime.fromisoformat(item_date)

                if item_date >= grace_cutoff:
                    # Within grace period
                    result["data_retained"] += 1

                    # Send notification if needed
                    days_until_action = (grace_cutoff - item_date).days
                    if days_until_action <= policy.notification_days_before:
                        await self._send_retention_notification(policy, item, days_until_action)
                else:
                    # Apply retention action
                    if not dry_run:
                        success = await self._apply_retention_action(
                            policy.action,
                            policy.data_category,
                            item
                        )
                        if success:
                            result["data_actioned"] += 1
                    else:
                        result["data_actioned"] += 1

            result["applied"] = True

        except Exception as e:
            logger.error(f"Failed to apply policy {policy.policy_id}: {e}")
            result["errors"].append(str(e))

        return result

    async def _get_data_items(
        self,
        category: DataCategory,
        cutoff_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get data items older than cutoff date for a category"""
        items = []

        if category == DataCategory.AUDIT_LOGS:
            # Read from audit log file
            audit_logger = get_audit_logger()
            events = await audit_logger.query_audit_events(
                end_time=cutoff_date,
                limit=10000
            )
            items = events

        elif category == DataCategory.SESSION_DATA:
            # Would query session storage
            # For now, return empty as implementation depends on session storage
            pass

        elif category == DataCategory.SECURITY_EVENTS:
            # Would query security event storage
            pass

        # Add other category handlers as needed

        return items

    async def _apply_retention_action(
        self,
        action: RetentionAction,
        category: DataCategory,
        item: Dict[str, Any]
    ) -> bool:
        """Apply retention action to a data item"""
        try:
            if action == RetentionAction.DELETE:
                return await self._delete_item(category, item)

            elif action == RetentionAction.ARCHIVE:
                return await self._archive_item(category, item)

            elif action == RetentionAction.ANONYMIZE:
                return await self._anonymize_item(category, item)

            elif action == RetentionAction.COMPRESS:
                return await self._compress_item(category, item)

            elif action == RetentionAction.MOVE_COLD_STORAGE:
                return await self._move_to_cold_storage(category, item)

            return False

        except Exception as e:
            logger.error(f"Failed to apply retention action {action}: {e}")
            return False

    async def _delete_item(self, category: DataCategory, item: Dict[str, Any]) -> bool:
        """Delete a data item"""
        # Implementation depends on storage mechanism
        logger.info(f"Would delete item from {category}: {item.get('id', item.get('event_id'))}")
        return True

    async def _archive_item(self, category: DataCategory, item: Dict[str, Any]) -> bool:
        """Archive a data item"""
        # Move to archive storage
        archive_path = Path(f"archives/{category.value}/{datetime.utcnow().year}")
        archive_path.mkdir(parents=True, exist_ok=True)

        archive_file = archive_path / f"{item.get('id', item.get('event_id'))}.json"

        with open(archive_file, 'w') as f:
            json.dump(item, f)

        logger.info(f"Archived item to {archive_file}")
        return True

    async def _anonymize_item(self, category: DataCategory, item: Dict[str, Any]) -> bool:
        """Anonymize personal data in item"""
        # Remove or hash PII fields
        pii_fields = ["user_id", "email", "name", "ip_address", "user_agent"]

        for field in pii_fields:
            if field in item:
                item[field] = hashlib.sha256(str(item[field]).encode()).hexdigest()[:16]

        logger.info(f"Anonymized item: {item.get('id', item.get('event_id'))}")
        return True

    async def _compress_item(self, category: DataCategory, item: Dict[str, Any]) -> bool:
        """Compress data item"""
        # Implementation would compress and store
        logger.info(f"Would compress item from {category}")
        return True

    async def _move_to_cold_storage(self, category: DataCategory, item: Dict[str, Any]) -> bool:
        """Move item to cold storage"""
        # Implementation would move to S3 Glacier or similar
        logger.info(f"Would move item to cold storage: {category}")
        return True

    async def _send_retention_notification(
        self,
        policy: RetentionPolicy,
        item: Dict[str, Any],
        days_remaining: int
    ):
        """Send notification about upcoming retention action"""
        await report_security_event(
            EventType.POLICY_VIOLATION,  # Using as notification
            ThreatLevel.INFO,
            f"Data retention action pending in {days_remaining} days",
            details={
                "policy_id": policy.policy_id,
                "data_category": policy.data_category.value,
                "action": policy.action.value,
                "item_id": item.get("id", item.get("event_id"))
            }
        )

    async def _log_execution(self, results: Dict[str, Any]):
        """Log retention policy execution"""
        self.execution_log.parent.mkdir(parents=True, exist_ok=True)

        with open(self.execution_log, 'a') as f:
            f.write(json.dumps(results) + '\n')

    def validate_compliance_requirements(self) -> Dict[str, List[str]]:
        """Validate that retention policies meet compliance requirements"""
        violations = {}

        # SOX Requirements
        sox_audit_policies = self.get_policies_by_framework("SOX")
        audit_policies = [p for p in sox_audit_policies if p.data_category == DataCategory.AUDIT_LOGS]

        if not audit_policies or all(p.retention_days < 2555 for p in audit_policies):
            violations["SOX"] = ["Audit logs must be retained for at least 7 years"]

        # GDPR Requirements
        gdpr_policies = self.get_policies_by_framework("GDPR")
        if not any(p.action == RetentionAction.DELETE or p.action == RetentionAction.ANONYMIZE
                  for p in gdpr_policies):
            violations["GDPR"] = ["GDPR requires deletion or anonymization capabilities"]

        # Add other compliance checks as needed

        return violations


# Global instance
_retention_manager: Optional[DataRetentionManager] = None

def get_retention_manager() -> DataRetentionManager:
    """Get global retention manager instance"""
    global _retention_manager
    if _retention_manager is None:
        _retention_manager = DataRetentionManager()
    return _retention_manager


# Export main classes and functions
__all__ = [
    'DataRetentionManager',
    'RetentionPolicy',
    'DataCategory',
    'RetentionAction',
    'get_retention_manager'
]

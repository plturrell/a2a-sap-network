"""
A2A Message Validation and Schema Enforcement
Provides comprehensive validation for A2A messages with schema enforcement and security checks
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
from jsonschema import validate, ValidationError

from .a2aTypes import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MessageValidationError(Exception):
    """Exception raised during message validation"""


@dataclass
class ValidationIssue:
    """Represents a validation issue"""

    severity: ValidationSeverity
    code: str
    message: str
    field_path: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of message validation"""

    is_valid: bool
    issues: List[ValidationIssue]
    validation_time_ms: float
    message_size_bytes: int
    schema_version: str

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


class A2AMessageValidator:
    """Comprehensive A2A message validator with schema enforcement"""

    SCHEMA_VERSION = "1.0"
    MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PARTS_COUNT = 1000
    MAX_STRING_LENGTH = 100000
    MAX_NESTING_DEPTH = 10

    def __init__(self, strict_mode: bool = False):
        """
        Initialize message validator

        Args:
            strict_mode: Enable strict validation (fails on warnings)
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "messages_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0,
            "average_validation_time_ms": 0.0,
            "schema_violations": 0,
            "security_issues": 0,
        }

        # Load validation schemas
        self.schemas = self._load_schemas()

        # Compile regex patterns for performance
        self._compiled_patterns = {
            "message_id": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "context_id": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "task_id": re.compile(r"^[a-zA-Z0-9_-]+$"),
            "agent_id": re.compile(r"^[a-zA-Z0-9._-]+$"),
            "timestamp": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
            "suspicious_content": re.compile(
                r"(script|javascript|eval|exec|system|shell)", re.IGNORECASE
            ),
        }

        # Security blacklist
        self.security_blacklist = {
            "scripts": ["<script", "javascript:", "eval(", "exec(", "system("],
            "sql_injection": ["'; drop", "union select", "or 1=1", "' or '1'='1"],
            "path_traversal": ["../", "..\\", "/etc/passwd", "c:\\windows"],
            "command_injection": ["; rm -rf", "| rm", "&& rm", "$(", "`"],
        }

    def validate_message(self, message: A2AMessage) -> ValidationResult:
        """
        Validate A2A message comprehensively

        Args:
            message: A2A message to validate

        Returns:
            ValidationResult with validation details
        """
        start_time = datetime.utcnow()
        issues = []

        try:
            # Convert to dict for size calculation
            message_dict = self._message_to_dict(message)
            message_json = json.dumps(message_dict)
            message_size = len(message_json.encode("utf-8"))

            # Basic structure validation
            issues.extend(self._validate_structure(message))

            # Schema validation
            issues.extend(self._validate_schema(message_dict))

            # Business logic validation
            issues.extend(self._validate_business_rules(message))

            # Security validation
            issues.extend(self._validate_security(message))

            # Size and limits validation
            issues.extend(self._validate_limits(message, message_size))

            # Protocol compliance validation
            issues.extend(self._validate_protocol_compliance(message))

            # Update statistics
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(issues, validation_time)

            # Determine if valid
            has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
            has_warnings = any(issue.severity == ValidationSeverity.WARNING for issue in issues)

            is_valid = not has_errors and (not self.strict_mode or not has_warnings)

            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                validation_time_ms=validation_time,
                message_size_bytes=message_size,
                schema_version=self.SCHEMA_VERSION,
            )

        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="VALIDATION_EXCEPTION",
                    message=f"Validation failed with exception: {str(e)}",
                )
            )

            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_stats(issues, validation_time)

            return ValidationResult(
                is_valid=False,
                issues=issues,
                validation_time_ms=validation_time,
                message_size_bytes=0,
                schema_version=self.SCHEMA_VERSION,
            )

    def _validate_structure(self, message: A2AMessage) -> List[ValidationIssue]:
        """Validate basic message structure"""
        issues = []

        # Required fields
        if not message.messageId:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_MESSAGE_ID",
                    message="Message ID is required",
                    field_path="messageId",
                )
            )
        elif not self._compiled_patterns["message_id"].match(message.messageId):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_MESSAGE_ID_FORMAT",
                    message="Message ID contains invalid characters",
                    field_path="messageId",
                    suggested_fix="Use only alphanumeric characters, hyphens, and underscores",
                )
            )

        # Role validation
        if not isinstance(message.role, MessageRole):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_ROLE",
                    message=f"Invalid message role: {message.role}",
                    field_path="role",
                )
            )

        # Parts validation
        if not message.parts:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EMPTY_PARTS",
                    message="Message has no parts",
                    field_path="parts",
                )
            )
        elif len(message.parts) > self.MAX_PARTS_COUNT:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="TOO_MANY_PARTS",
                    message=f"Message has {len(message.parts)} parts, maximum allowed is {self.MAX_PARTS_COUNT}",
                    field_path="parts",
                )
            )

        # Validate each part
        for i, part in enumerate(message.parts):
            part_issues = self._validate_message_part(part, f"parts[{i}]")
            issues.extend(part_issues)

        return issues

    def _validate_message_part(self, part: MessagePart, field_path: str) -> List[ValidationIssue]:
        """Validate individual message part"""
        issues = []

        # Kind validation
        if not part.kind:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_PART_KIND",
                    message="Message part kind is required",
                    field_path=f"{field_path}.kind",
                )
            )

        # Content validation
        has_content = any([part.text, part.data, part.file])
        if not has_content:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="EMPTY_PART_CONTENT",
                    message="Message part has no content",
                    field_path=field_path,
                )
            )

        # Text length validation
        if part.text and len(part.text) > self.MAX_STRING_LENGTH:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="TEXT_TOO_LONG",
                    message=f"Text content exceeds maximum length of {self.MAX_STRING_LENGTH}",
                    field_path=f"{field_path}.text",
                )
            )

        # Data structure validation
        if part.data:
            data_issues = self._validate_nested_data(part.data, f"{field_path}.data")
            issues.extend(data_issues)

        return issues

    def _validate_nested_data(
        self, data: Any, field_path: str, depth: int = 0
    ) -> List[ValidationIssue]:
        """Validate nested data structures"""
        issues = []

        if depth > self.MAX_NESTING_DEPTH:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="EXCESSIVE_NESTING",
                    message=f"Data nesting exceeds maximum depth of {self.MAX_NESTING_DEPTH}",
                    field_path=field_path,
                )
            )
            return issues

        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, str):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="NON_STRING_KEY",
                            message="Dictionary keys must be strings",
                            field_path=f"{field_path}.{key}",
                        )
                    )

                if isinstance(value, (dict, list)):
                    nested_issues = self._validate_nested_data(
                        value, f"{field_path}.{key}", depth + 1
                    )
                    issues.extend(nested_issues)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    nested_issues = self._validate_nested_data(
                        item, f"{field_path}[{i}]", depth + 1
                    )
                    issues.extend(nested_issues)

        return issues

    def _validate_schema(self, message_dict: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate against JSON schema"""
        issues = []

        try:
            validate(instance=message_dict, schema=self.schemas["a2a_message"])
        except ValidationError as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SCHEMA_VALIDATION_FAILED",
                    message=f"Schema validation failed: {e.message}",
                    field_path=".".join(str(x) for x in e.path) if e.path else None,
                )
            )
            self.validation_stats["schema_violations"] += 1

        return issues

    def _validate_business_rules(self, message: A2AMessage) -> List[ValidationIssue]:
        """Validate business logic rules"""
        issues = []

        # Timestamp validation
        if message.timestamp:
            try:
                timestamp = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00"))
                now = datetime.utcnow()

                # Check if timestamp is too far in the future
                if timestamp > now + timedelta(minutes=5):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="FUTURE_TIMESTAMP",
                            message="Message timestamp is in the future",
                            field_path="timestamp",
                        )
                    )

                # Check if timestamp is too old
                if timestamp < now - timedelta(days=1):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="OLD_TIMESTAMP",
                            message="Message timestamp is more than 1 day old",
                            field_path="timestamp",
                        )
                    )

            except ValueError:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_TIMESTAMP_FORMAT",
                        message="Invalid timestamp format",
                        field_path="timestamp",
                    )
                )

        # Context and task ID consistency
        if message.contextId and message.taskId:
            if not self._compiled_patterns["context_id"].match(message.contextId):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="INVALID_CONTEXT_ID_FORMAT",
                        message="Context ID format is not standard",
                        field_path="contextId",
                    )
                )

            if not self._compiled_patterns["task_id"].match(message.taskId):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="INVALID_TASK_ID_FORMAT",
                        message="Task ID format is not standard",
                        field_path="taskId",
                    )
                )

        return issues

    def _validate_security(self, message: A2AMessage) -> List[ValidationIssue]:
        """Validate security aspects of the message"""
        issues = []

        # Check all text content for security issues
        text_contents = []

        # Add all text from parts
        for part in message.parts:
            if part.text:
                text_contents.append(part.text)
            if part.data and isinstance(part.data, dict):
                text_contents.extend(self._extract_text_from_data(part.data))

        # Security checks
        for text in text_contents:
            security_issues = self._check_security_threats(text)
            issues.extend(security_issues)

        # Signature validation
        if message.signature:
            if len(message.signature) < 32:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="WEAK_SIGNATURE",
                        message="Message signature appears to be weak",
                        field_path="signature",
                    )
                )

        return issues

    def _check_security_threats(self, text: str) -> List[ValidationIssue]:
        """Check text content for security threats"""
        issues = []

        for threat_type, patterns in self.security_blacklist.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code=f"SECURITY_THREAT_{threat_type.upper()}",
                            message=f"Potential {threat_type.replace('_', ' ')} detected: '{pattern}'",
                            suggested_fix="Remove or sanitize suspicious content",
                        )
                    )
                    self.validation_stats["security_issues"] += 1

        # Check for suspicious patterns
        if self._compiled_patterns["suspicious_content"].search(text):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="SUSPICIOUS_CONTENT",
                    message="Content contains potentially suspicious keywords",
                )
            )

        return issues

    def _extract_text_from_data(self, data: Any) -> List[str]:
        """Extract all string values from nested data structures"""
        text_values = []

        if isinstance(data, str):
            text_values.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                text_values.extend(self._extract_text_from_data(value))
        elif isinstance(data, list):
            for item in data:
                text_values.extend(self._extract_text_from_data(item))

        return text_values

    def _validate_limits(self, message: A2AMessage, message_size: int) -> List[ValidationIssue]:
        """Validate size and count limits"""
        issues = []

        # Message size validation
        if message_size > self.MAX_MESSAGE_SIZE:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MESSAGE_TOO_LARGE",
                    message=f"Message size {message_size} bytes exceeds maximum of {self.MAX_MESSAGE_SIZE} bytes",
                    suggested_fix="Reduce message content or use file attachments",
                )
            )
        elif message_size > self.MAX_MESSAGE_SIZE * 0.8:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="MESSAGE_SIZE_WARNING",
                    message=f"Message size {message_size} bytes is approaching the limit",
                )
            )

        return issues

    def _validate_protocol_compliance(self, message: A2AMessage) -> List[ValidationIssue]:
        """Validate A2A protocol compliance"""
        issues = []

        # Check for required protocol fields based on role
        if message.role == MessageRole.AGENT:
            if not message.contextId:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MISSING_CONTEXT_ID",
                        message="Agent messages should include contextId for traceability",
                        field_path="contextId",
                    )
                )

        # Check message part kinds
        valid_kinds = {"text", "data", "file", "image", "audio", "video", "tool_use", "tool_result"}
        for i, part in enumerate(message.parts):
            if part.kind not in valid_kinds:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="NON_STANDARD_PART_KIND",
                        message=f"Non-standard part kind: {part.kind}",
                        field_path=f"parts[{i}].kind",
                        suggested_fix=f"Use one of: {', '.join(valid_kinds)}",
                    )
                )

        return issues

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for validation"""
        return {
            "a2a_message": {
                "type": "object",
                "required": ["messageId", "role", "parts"],
                "properties": {
                    "messageId": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "enum": ["user", "agent", "system"]},
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["kind"],
                            "properties": {
                                "kind": {"type": "string"},
                                "text": {"type": ["string", "null"]},
                                "data": {"type": ["object", "null"]},
                                "file": {"type": ["object", "null"]},
                            },
                        },
                    },
                    "taskId": {"type": ["string", "null"]},
                    "contextId": {"type": ["string", "null"]},
                    "timestamp": {"type": ["string", "null"]},
                    "signature": {"type": ["string", "null"]},
                },
            }
        }

    def _message_to_dict(self, message: A2AMessage) -> Dict[str, Any]:
        """Convert A2AMessage to dictionary"""
        return {
            "messageId": message.messageId,
            "role": message.role.value,
            "parts": [
                {"kind": part.kind, "text": part.text, "data": part.data, "file": part.file}
                for part in message.parts
            ],
            "taskId": message.taskId,
            "contextId": message.contextId,
            "timestamp": message.timestamp,
            "signature": message.signature,
        }

    def _update_stats(self, issues: List[ValidationIssue], validation_time: float) -> None:
        """Update validation statistics"""
        self.validation_stats["messages_validated"] += 1

        error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)

        self.validation_stats["validation_errors"] += error_count
        self.validation_stats["validation_warnings"] += warning_count

        # Update average validation time
        total_messages = self.validation_stats["messages_validated"]
        current_avg = self.validation_stats["average_validation_time_ms"]
        self.validation_stats["average_validation_time_ms"] = (
            current_avg * (total_messages - 1) + validation_time
        ) / total_messages

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()

    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        for key in self.validation_stats:
            self.validation_stats[key] = 0


# Global validator instance
_default_validator = None


def get_message_validator(strict_mode: bool = False) -> A2AMessageValidator:
    """Get global message validator instance"""
    global _default_validator

    if _default_validator is None:
        _default_validator = A2AMessageValidator(strict_mode=strict_mode)

    return _default_validator


# Convenience functions
def validate_message(message: A2AMessage, strict_mode: bool = False) -> ValidationResult:
    """Validate A2A message using global validator"""
    validator = get_message_validator(strict_mode)
    return validator.validate_message(message)


def is_message_valid(message: A2AMessage, strict_mode: bool = False) -> bool:
    """Check if A2A message is valid"""
    result = validate_message(message, strict_mode)
    return result.is_valid

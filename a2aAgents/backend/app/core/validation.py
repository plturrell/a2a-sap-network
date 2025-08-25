"""
Input Validation Framework for A2A API Security
Provides comprehensive input validation and sanitization
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, validator, ValidationError
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Reject any suspicious input
    MODERATE = "moderate"  # Sanitize and warn
    PERMISSIVE = "permissive"  # Log only


class InputSanitizer:
    """Secure input sanitization utilities"""

    @staticmethod
    def sanitize_string(
        value: str,
        max_length: int = 1000,
        allow_html: bool = False,
        allow_sql_chars: bool = False
    ) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Length check
        if len(value) > max_length:
            raise ValueError(f"String too long. Maximum length: {max_length}")

        # Remove null bytes
        value = value.replace('\x00', '')

        # HTML sanitization
        if not allow_html:
            # Basic HTML entity encoding for dangerous characters
            html_entities = {
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;',
                '&': '&amp;',
            }
            for char, entity in html_entities.items():
                value = value.replace(char, entity)

        # SQL injection prevention
        if not allow_sql_chars:
            dangerous_sql_patterns = [
                r"(?i)(union\s+select)",
                r"(?i)(insert\s+into)",
                r"(?i)(delete\s+from)",
                r"(?i)(drop\s+table)",
                r"(?i)(alter\s+table)",
                r"(?i)(create\s+table)",
                r"(?i)(exec\s*\()",
                r"(?i)(execute\s*\()",
                r"--",
                r"/\*",
                r"\*/",
            ]

            for pattern in dangerous_sql_patterns:
                if re.search(pattern, value):
                    raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")

        return value.strip()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")

        # Remove path traversal patterns
        filename = re.sub(r'\.\.[\\/]', '', filename)
        filename = re.sub(r'^[\\/]+', '', filename)

        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)

        # Limit length
        if len(filename) > 255:
            raise ValueError("Filename too long")

        if not filename or filename in ['.', '..']:
            raise ValueError("Invalid filename")

        return filename

    @staticmethod
    def validate_json_structure(
        data: Dict[str, Any],
        max_depth: int = 10,
        max_keys: int = 1000
    ) -> Dict[str, Any]:
        """Validate JSON structure for security"""

        def count_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValueError(f"JSON structure too deep. Maximum depth: {max_depth}")

            if isinstance(obj, dict):
                if len(obj) > max_keys:
                    raise ValueError(f"Too many keys in JSON object. Maximum: {max_keys}")
                for value in obj.values():
                    count_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                if len(obj) > max_keys:
                    raise ValueError(f"Too many items in JSON array. Maximum: {max_keys}")
                for item in obj:
                    count_depth(item, current_depth + 1)

        count_depth(data)
        return data


class APIValidator:
    """API endpoint input validator"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.sanitizer = InputSanitizer()

    def validate_agent_id(self, agent_id: str) -> str:
        """Validate A2A agent ID format"""
        if not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a string")

        # Agent ID should follow specific pattern
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', agent_id):
            raise ValueError("Invalid agent ID format. Use only alphanumeric characters, hyphens, and underscores")

        return agent_id

    def validate_workflow_id(self, workflow_id: str) -> str:
        """Validate workflow ID"""
        if not isinstance(workflow_id, str):
            raise ValueError("Workflow ID must be a string")

        if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', workflow_id):
            raise ValueError("Invalid workflow ID format")

        return workflow_id

    def validate_message_content(self, content: str) -> str:
        """Validate message content"""
        return self.sanitizer.sanitize_string(
            content,
            max_length=10000,
            allow_html=False,
            allow_sql_chars=False
        )

    def validate_file_upload(self, filename: str, file_size: int, content_type: str) -> Dict[str, Any]:
        """Validate file upload parameters"""
        # Validate filename
        safe_filename = self.sanitizer.sanitize_filename(filename)

        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            raise ValueError(f"File too large. Maximum size: {max_size} bytes")

        # Validate content type whitelist
        allowed_types = {
            'application/json',
            'application/xml',
            'text/plain',
            'text/csv',
            'application/pdf',
            'image/jpeg',
            'image/png',
            'image/gif'
        }

        if content_type not in allowed_types:
            raise ValueError(f"File type not allowed: {content_type}")

        # Check for executable extensions
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.js', '.jar'}
        file_ext = '.' + safe_filename.split('.')[-1].lower() if '.' in safe_filename else ''

        if file_ext in dangerous_extensions:
            raise ValueError(f"Dangerous file extension not allowed: {file_ext}")

        return {
            "filename": safe_filename,
            "size": file_size,
            "content_type": content_type,
            "validated": True
        }

    def validate_database_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate database query parameters"""
        validated_params = {}

        for key, value in params.items():
            # Validate parameter name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid parameter name: {key}")

            # Validate parameter value
            if isinstance(value, str):
                validated_params[key] = self.sanitizer.sanitize_string(
                    value,
                    max_length=1000,
                    allow_sql_chars=False
                )
            elif isinstance(value, (int, float, bool)):
                validated_params[key] = value
            elif isinstance(value, list):
                # Validate list items
                validated_list = []
                for item in value[:100]:  # Limit list size
                    if isinstance(item, str):
                        validated_list.append(
                            self.sanitizer.sanitize_string(item, max_length=100)
                        )
                    elif isinstance(item, (int, float, bool)):
                        validated_list.append(item)
                    else:
                        raise ValueError(f"Invalid list item type in parameter: {key}")
                validated_params[key] = validated_list
            else:
                raise ValueError(f"Invalid parameter type for: {key}")

        return validated_params

    def validate_api_request_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API request body"""
        # Check JSON structure
        self.sanitizer.validate_json_structure(body, max_depth=10, max_keys=100)

        # Recursively sanitize string values
        def sanitize_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: sanitize_recursive(v)
                    for k, v in obj.items()
                    if isinstance(k, str) and len(k) <= 100
                }
            elif isinstance(obj, list):
                return [sanitize_recursive(item) for item in obj[:100]]  # Limit array size
            elif isinstance(obj, str):
                return self.sanitizer.sanitize_string(obj, max_length=10000)
            elif isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            else:
                raise ValueError(f"Invalid data type: {type(obj)}")

        return sanitize_recursive(body)


class SecurityMiddleware:
    """Security validation middleware for API endpoints"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validator = APIValidator(validation_level)

    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request data"""
        try:
            validated_data = {}

            # Validate common fields
            if 'agent_id' in request_data:
                validated_data['agent_id'] = self.validator.validate_agent_id(
                    request_data['agent_id']
                )

            if 'workflow_id' in request_data:
                validated_data['workflow_id'] = self.validator.validate_workflow_id(
                    request_data['workflow_id']
                )

            if 'message' in request_data:
                validated_data['message'] = self.validator.validate_message_content(
                    request_data['message']
                )

            if 'query_params' in request_data:
                validated_data['query_params'] = self.validator.validate_database_query_params(
                    request_data['query_params']
                )

            # Validate request body if present
            if 'body' in request_data and isinstance(request_data['body'], dict):
                validated_data['body'] = self.validator.validate_api_request_body(
                    request_data['body']
                )

            # Copy over other safe fields
            safe_fields = {'user_id', 'timestamp', 'request_id', 'correlation_id'}
            for field in safe_fields:
                if field in request_data:
                    validated_data[field] = request_data[field]

            return validated_data

        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input validation failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal validation error"
            )


# Pydantic models for common API inputs
class AgentMessageRequest(BaseModel):
    """Secure model for agent messages"""
    agent_id: str
    message_type: str
    content: str
    target_agent_id: Optional[str] = None
    correlation_id: Optional[str] = None

    @validator('agent_id', 'target_agent_id')
    def validate_agent_ids(cls, v):
        if v is not None and not re.match(r'^[a-zA-Z0-9_-]{1,64}$', v):
            raise ValueError('Invalid agent ID format')
        return v

    @validator('message_type')
    def validate_message_type(cls, v):
        allowed_types = {
            'request', 'response', 'notification', 'error',
            'data_transfer', 'health_check', 'status_update'
        }
        if v not in allowed_types:
            raise ValueError(f'Invalid message type. Allowed: {allowed_types}')
        return v

    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10000:
            raise ValueError('Message content too long')
        # Basic sanitization
        if any(char in v for char in ['<script', '<iframe', 'javascript:']):
            raise ValueError('Potentially dangerous content detected')
        return v


class WorkflowExecutionRequest(BaseModel):
    """Secure model for workflow execution"""
    workflow_id: str
    variables: Dict[str, Any] = {}
    initiator_agent_id: str

    @validator('workflow_id')
    def validate_workflow_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', v):
            raise ValueError('Invalid workflow ID format')
        return v

    @validator('variables')
    def validate_variables(cls, v):
        # Limit number of variables
        if len(v) > 50:
            raise ValueError('Too many variables')

        # Validate each variable
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError(f'Invalid variable name: {key}')

            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f'Variable value too long: {key}')

        return v


# Export validation utilities
__all__ = [
    'ValidationLevel',
    'InputSanitizer',
    'APIValidator',
    'SecurityMiddleware',
    'AgentMessageRequest',
    'WorkflowExecutionRequest'
]
"""
Utility functions for A2A Agent SDK
"""

import hashlib
import hmac
import json
import secrets
import string
from typing import Dict, Any, Optional
from datetime import datetime
import re

from .types import A2AMessage


def create_agent_id(name: str, organization: str = "a2a") -> str:
    """
    Create standardized agent ID

    Args:
        name: Agent name
        organization: Organization name

    Returns:
        Formatted agent ID
    """
    # Clean and format name
    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    clean_name = re.sub(r'\s+', '_', clean_name.strip()).lower()

    # Clean organization
    clean_org = re.sub(r'[^a-zA-Z0-9]', '', organization).lower()

    return f"{clean_org}_{clean_name}"


def validate_message(message: A2AMessage) -> Dict[str, Any]:
    """
    Validate A2A message structure

    Args:
        message: A2A message to validate

    Returns:
        Validation result with success flag and errors
    """
    errors = []

    # Check required fields
    if not message.messageId:
        errors.append("messageId is required")

    if not message.role:
        errors.append("role is required")

    if not message.parts:
        errors.append("parts array cannot be empty")

    # Validate parts
    for i, part in enumerate(message.parts):
        if not part.kind:
            errors.append(f"part[{i}].kind is required")

        # Check that at least one content field is present
        if not any([part.text, part.data, part.file]):
            errors.append(f"part[{i}] must have at least one of: text, data, file")

    # Validate timestamp format
    if message.timestamp:
        try:
            datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))
        except ValueError:
            errors.append("timestamp must be in ISO format")

    return {
        "success": len(errors) == 0,
        "errors": errors
    }


def sign_message(message: A2AMessage, secret_key: str) -> str:
    """
    Sign A2A message for integrity verification

    Args:
        message: Message to sign
        secret_key: Secret key for signing

    Returns:
        Message signature
    """
    # Create canonical representation
    message_dict = message.model_dump()

    # Remove signature if present
    if "signature" in message_dict:
        del message_dict["signature"]

    # Sort keys for consistent representation
    canonical = json.dumps(message_dict, sort_keys=True, separators=(',', ':'))

    # Create HMAC signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        canonical.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return signature


def verify_message_signature(message: A2AMessage, secret_key: str) -> bool:
    """
    Verify A2A message signature

    Args:
        message: Message with signature
        secret_key: Secret key for verification

    Returns:
        True if signature is valid
    """
    if not message.signature:
        return False

    expected_signature = sign_message(message, secret_key)

    # Use constant time comparison to prevent timing attacks
    return hmac.compare_digest(message.signature, expected_signature)


def generate_api_key(prefix: str = "a2a", length: int = 32) -> str:
    """
    Generate secure API key

    Args:
        prefix: Key prefix
        length: Key length (excluding prefix)

    Returns:
        Generated API key
    """
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}_{key}"


def hash_data(data: Any, algorithm: str = "sha256") -> str:
    """
    Hash data for integrity checking

    Args:
        data: Data to hash
        algorithm: Hash algorithm

    Returns:
        Hash digest as hex string
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        data_bytes = canonical.encode('utf-8')
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = str(data).encode('utf-8')

    if algorithm == "sha256":
        return hashlib.sha256(data_bytes).hexdigest()
    elif algorithm == "md5":
        # MD5 is deprecated - use SHA256 instead
        import warnings
        warnings.warn("MD5 is cryptographically weak, consider using SHA256", DeprecationWarning)
        return hashlib.md5(data_bytes).hexdigest()
    elif algorithm == "sha1":
        # SHA1 is deprecated - use SHA256 instead
        import warnings
        warnings.warn("SHA1 is cryptographically weak, consider using SHA256", DeprecationWarning)
        return hashlib.sha1(data_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def extract_message_data(message: A2AMessage) -> Dict[str, Any]:
    """
    Extract data payload from message parts

    Args:
        message: A2A message

    Returns:
        Combined data from all data parts
    """
    combined_data = {}

    for part in message.parts:
        if part.kind == "data" and part.data:
            combined_data.update(part.data)

    return combined_data


def create_error_response(
    error_code: int,
    error_message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized error response

    Args:
        error_code: Error code
        error_message: Error message
        details: Optional error details

    Returns:
        Error response
    """
    response = {
        "success": False,
        "error": {
            "code": error_code,
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    if details:
        response["error"]["details"] = details

    return response


def create_success_response(
    result: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized success response

    Args:
        result: Result data
        metadata: Optional metadata

    Returns:
        Success response
    """
    response = {
        "success": True,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }

    if metadata:
        response["metadata"] = metadata

    return response


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem storage

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')

    return sanitized


def parse_endpoint_url(url: str) -> Dict[str, str]:
    """
    Parse endpoint URL into components

    Args:
        url: URL to parse

    Returns:
        URL components
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(url)

    return {
        "scheme": parsed.scheme,
        "hostname": parsed.hostname,
        "port": str(parsed.port) if parsed.port else ("443" if parsed.scheme == "https" else "80"),
        "path": parsed.path,
        "query": parsed.query,
        "full_url": url
    }
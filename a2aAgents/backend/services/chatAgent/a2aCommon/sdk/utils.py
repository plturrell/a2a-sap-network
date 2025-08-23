"""
A2A SDK utilities
"""

from typing import Dict, Any

def create_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Create a success response"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "error": None
    }

def create_error_response(code: int, message: str, details: Any = None) -> Dict[str, Any]:
    """Create an error response"""
    return {
        "success": False,
        "message": message,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "details": details
        }
    }
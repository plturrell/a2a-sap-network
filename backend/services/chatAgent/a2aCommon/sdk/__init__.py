"""
A2A SDK utilities - Re-exports from main SDK
"""

try:
    # Try to import from main SDK
    import sys
    from pathlib import Path
    
    sdk_path = Path(__file__).parent.parent.parent.parent / "app" / "a2a" / "sdk"
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    
    from utils import create_success_response, create_error_response
    
except ImportError:
    # Fallback implementations
    from typing import Dict, Any


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    
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

__all__ = ['create_success_response', 'create_error_response']
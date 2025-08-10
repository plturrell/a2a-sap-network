"""
A2A Network Validation Utilities

Validation functions for addresses, configurations, and parameters.
"""

def validate_address(address):
    """Validate Ethereum address"""
    if not address or not isinstance(address, str):
        return False
    return address.startswith('0x') and len(address) == 42

def validate_config(config):
    """Validate client configuration"""
    if not isinstance(config, dict):
        return {'is_valid': False, 'errors': ['Configuration must be a dictionary']}
    
    errors = []
    required_fields = []  # No required fields for now
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    return {'is_valid': len(errors) == 0, 'errors': errors}

def validate_agent_params(params):
    """Validate agent parameters"""
    if not isinstance(params, dict):
        return {'is_valid': False, 'errors': ['Parameters must be a dictionary']}
    
    required_fields = ['name', 'description', 'endpoint', 'capabilities']
    errors = []
    
    for field in required_fields:
        if field not in params:
            errors.append(f"Missing required field: {field}")
    
    return {'is_valid': len(errors) == 0, 'errors': errors}

__all__ = ['validate_address', 'validate_config', 'validate_agent_params']
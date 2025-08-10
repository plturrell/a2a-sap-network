"""
A2A Network Formatting Utilities

Functions for formatting data returned from contracts and APIs.
"""

def format_agent_data(agent_id, raw_data):
    """Format agent data from contract"""
    return {
        'id': agent_id,
        'name': raw_data.get('name', 'Unknown Agent'),
        'description': raw_data.get('description', ''),
        'endpoint': raw_data.get('endpoint', ''),
        'owner': raw_data.get('owner', '0x0000000000000000000000000000000000000000'),
        'capabilities': raw_data.get('capabilities', []),
        'message_count': raw_data.get('message_count', 0),
        'metadata': raw_data.get('metadata', '{}'),
        'active': raw_data.get('active', True),
        'reputation': raw_data.get('reputation', 0)
    }

def parse_agent_capabilities(capabilities):
    """Parse agent capabilities from various formats"""
    if isinstance(capabilities, list):
        return capabilities
    elif isinstance(capabilities, str):
        try:
            import json
            return json.loads(capabilities)
        except:
            return capabilities.split(',')
    else:
        return []

__all__ = ['format_agent_data', 'parse_agent_capabilities']
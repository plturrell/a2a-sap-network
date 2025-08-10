"""
A2A Network Utilities

Utility modules for validation, formatting, error handling, and cryptographic operations.
"""

# Placeholder modules for missing utilities
class ErrorCode:
    """Error code constants"""
    INVALID_CONFIG = "INVALID_CONFIG"
    UNSUPPORTED_NETWORK = "UNSUPPORTED_NETWORK"
    NO_PROVIDER = "NO_PROVIDER"
    INVALID_PRIVATE_KEY = "INVALID_PRIVATE_KEY"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    WRONG_NETWORK = "WRONG_NETWORK"
    DISCONNECTION_FAILED = "DISCONNECTION_FAILED"
    NO_CONTRACTS = "NO_CONTRACTS"
    INVALID_CONTRACT_ADDRESS = "INVALID_CONTRACT_ADDRESS"
    CONTRACT_INITIALIZATION_FAILED = "CONTRACT_INITIALIZATION_FAILED"
    CONTRACT_NOT_FOUND = "CONTRACT_NOT_FOUND"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"
    WEBSOCKET_NOT_CONNECTED = "WEBSOCKET_NOT_CONNECTED"
    NETWORK_INFO_FAILED = "NETWORK_INFO_FAILED"
    INVALID_PARAMS = "INVALID_PARAMS"
    NO_SIGNER = "NO_SIGNER"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    UPDATE_FAILED = "UPDATE_FAILED"
    FETCH_FAILED = "FETCH_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    STATUS_UPDATE_FAILED = "STATUS_UPDATE_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    DEREGISTRATION_FAILED = "DEREGISTRATION_FAILED"
    ESTIMATION_FAILED = "ESTIMATION_FAILED"
    INVALID_ADDRESS = "INVALID_ADDRESS"

class A2AError(Exception):
    """Base A2A Network error"""
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)

# Validation functions
import re
import json
from typing import Dict, List, Any, Optional

def validate_address(address: str) -> bool:
    """Validate Ethereum address"""
    if not address or not isinstance(address, str):
        return False
    
    # Check format: 0x + 40 hex characters
    if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
        return False
    
    # Could add checksum validation here if needed
    return True

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate client configuration"""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = {
        'network': 'Network identifier (e.g., mainnet, testnet)',
        'rpc_url': 'RPC endpoint URL'
    }
    
    for field, description in required_fields.items():
        if field not in config or not config[field]:
            errors.append(f"Missing required field '{field}': {description}")
    
    # Validate RPC URL format
    if 'rpc_url' in config:
        rpc_url = config['rpc_url']
        if not (rpc_url.startswith('http://') or rpc_url.startswith('https://') or rpc_url.startswith('ws://') or rpc_url.startswith('wss://')):
            errors.append("RPC URL must start with http://, https://, ws://, or wss://")
    
    # Validate network
    if 'network' in config:
        valid_networks = ['mainnet', 'testnet', 'devnet', 'local']
        if config['network'] not in valid_networks:
            warnings.append(f"Unusual network '{config['network']}'. Common values: {', '.join(valid_networks)}")
    
    # Optional fields validation
    if 'private_key' in config and config['private_key']:
        if not re.match(r'^0x[a-fA-F0-9]{64}$', config['private_key']):
            errors.append("Private key must be 0x followed by 64 hex characters")
    
    if 'contracts' in config and isinstance(config['contracts'], dict):
        for contract_name, address in config['contracts'].items():
            if not validate_address(address):
                errors.append(f"Invalid contract address for '{contract_name}': {address}")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_agent_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent parameters"""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['name', 'description', 'endpoint', 'capabilities']
    for field in required_fields:
        if field not in params or not params[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate name
    if 'name' in params:
        name = params['name']
        if len(name) < 3:
            errors.append("Agent name must be at least 3 characters long")
        if len(name) > 100:
            errors.append("Agent name must not exceed 100 characters")
        if not re.match(r'^[a-zA-Z0-9_\-\s]+$', name):
            errors.append("Agent name can only contain letters, numbers, spaces, hyphens, and underscores")
    
    # Validate description
    if 'description' in params:
        description = params['description']
        if len(description) < 10:
            errors.append("Description must be at least 10 characters long")
        if len(description) > 1000:
            errors.append("Description must not exceed 1000 characters")
    
    # Validate endpoint
    if 'endpoint' in params:
        endpoint = params['endpoint']
        if not re.match(r'^https?://', endpoint):
            errors.append("Endpoint must be a valid HTTP/HTTPS URL")
        if len(endpoint) > 500:
            errors.append("Endpoint URL must not exceed 500 characters")
    
    # Validate capabilities
    if 'capabilities' in params:
        capabilities = params['capabilities']
        if not isinstance(capabilities, list):
            errors.append("Capabilities must be a list")
        elif len(capabilities) == 0:
            warnings.append("No capabilities specified - agent may have limited functionality")
        else:
            for cap in capabilities:
                if not isinstance(cap, str) or len(cap) == 0:
                    errors.append("Each capability must be a non-empty string")
                    break
    
    # Optional fields
    if 'metadata' in params:
        metadata = params['metadata']
        if isinstance(metadata, str):
            try:
                json.loads(metadata)
            except json.JSONDecodeError:
                errors.append("Metadata must be valid JSON if provided as string")
        elif not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary or JSON string")
    
    if 'tags' in params:
        tags = params['tags']
        if not isinstance(tags, list):
            errors.append("Tags must be a list")
        else:
            for tag in tags:
                if not isinstance(tag, str) or not re.match(r'^[a-zA-Z0-9_\-]+$', tag):
                    errors.append("Tags must be alphanumeric strings with hyphens/underscores only")
                    break
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# Formatting functions
def format_agent_data(agent_id: str, raw_data: Any) -> Dict[str, Any]:
    """Format agent data from contract"""
    # Handle different data formats from contract
    if isinstance(raw_data, (list, tuple)) and len(raw_data) >= 6:
        # Assuming contract returns tuple/array format
        return {
            'id': agent_id,
            'name': raw_data[0] or 'Unnamed Agent',
            'description': raw_data[1] or 'No description provided',
            'endpoint': raw_data[2] or '',
            'owner': raw_data[3] or '0x0000000000000000000000000000000000000000',
            'message_count': int(raw_data[4]) if len(raw_data) > 4 else 0,
            'metadata': raw_data[5] if len(raw_data) > 5 else '{}',
            'is_active': bool(raw_data[6]) if len(raw_data) > 6 else True,
            'registration_time': int(raw_data[7]) if len(raw_data) > 7 else 0
        }
    elif isinstance(raw_data, dict):
        # Handle dictionary format
        return {
            'id': agent_id,
            'name': raw_data.get('name', 'Unnamed Agent'),
            'description': raw_data.get('description', 'No description provided'),
            'endpoint': raw_data.get('endpoint', ''),
            'owner': raw_data.get('owner', '0x0000000000000000000000000000000000000000'),
            'message_count': int(raw_data.get('messageCount', 0)),
            'metadata': raw_data.get('metadata', '{}'),
            'is_active': raw_data.get('isActive', True),
            'registration_time': int(raw_data.get('registrationTime', 0))
        }
    else:
        # Fallback for unknown formats
        return {
            'id': agent_id,
            'name': 'Unknown Agent',
            'description': 'Unable to parse agent data',
            'endpoint': '',
            'owner': '0x0000000000000000000000000000000000000000',
            'message_count': 0,
            'metadata': '{}',
            'is_active': False,
            'registration_time': 0
        }

def parse_agent_capabilities(capabilities: Any) -> List[str]:
    """Parse agent capabilities from various formats"""
    parsed_capabilities = []
    
    if isinstance(capabilities, str):
        # Try to parse as JSON array
        try:
            caps_data = json.loads(capabilities)
            if isinstance(caps_data, list):
                parsed_capabilities = [str(cap) for cap in caps_data if cap]
        except json.JSONDecodeError:
            # Try comma-separated format
            parsed_capabilities = [cap.strip() for cap in capabilities.split(',') if cap.strip()]
    
    elif isinstance(capabilities, list):
        # Already a list
        parsed_capabilities = [str(cap) for cap in capabilities if cap]
    
    elif isinstance(capabilities, dict):
        # Capability dictionary format
        for key, value in capabilities.items():
            if value:  # If capability is enabled
                parsed_capabilities.append(key)
    
    # Clean and validate capabilities
    valid_capabilities = []
    for cap in parsed_capabilities:
        # Normalize capability name
        normalized = cap.lower().strip().replace(' ', '_')
        if normalized and re.match(r'^[a-zA-Z0-9_\-]+$', normalized):
            valid_capabilities.append(normalized)
    
    return valid_capabilities

# Additional utility functions
def format_wei_to_ether(wei_amount: int) -> str:
    """Convert Wei to Ether with proper formatting"""
    ether = wei_amount / 10**18
    return f"{ether:.6f} ETH"

def parse_metadata(metadata: Any) -> Dict[str, Any]:
    """Parse metadata from various formats"""
    if isinstance(metadata, dict):
        return metadata
    elif isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {"raw": metadata}
    else:
        return {"raw": str(metadata)}

def validate_transaction_params(tx_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate transaction parameters"""
    errors = []
    
    if 'to' in tx_params:
        if not validate_address(tx_params['to']):
            errors.append("Invalid 'to' address")
    
    if 'from' in tx_params:
        if not validate_address(tx_params['from']):
            errors.append("Invalid 'from' address")
    
    if 'value' in tx_params:
        if not isinstance(tx_params['value'], (int, str)):
            errors.append("'value' must be an integer or hex string")
        elif isinstance(tx_params['value'], int) and tx_params['value'] < 0:
            errors.append("'value' cannot be negative")
    
    if 'gas' in tx_params:
        if not isinstance(tx_params['gas'], (int, str)):
            errors.append("'gas' must be an integer or hex string")
    
    if 'gasPrice' in tx_params:
        if not isinstance(tx_params['gasPrice'], (int, str)):
            errors.append("'gasPrice' must be an integer or hex string")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors
    }

# Create error module
class errors:
    A2AError = A2AError
    ErrorCode = ErrorCode

# Create validation module
class validation:
    validate_address = staticmethod(validate_address)
    validate_config = staticmethod(validate_config)
    validate_agent_params = staticmethod(validate_agent_params)

# Create formatting module
class formatting:
    format_agent_data = staticmethod(format_agent_data)
    parse_agent_capabilities = staticmethod(parse_agent_capabilities)

# Cryptographic utilities
import hashlib
import hmac
import secrets
from typing import Tuple

class crypto:
    """Cryptographic utilities for A2A Network"""
    
    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """Generate a new private/public key pair"""
        # Generate 32 bytes of randomness for private key
        private_key_bytes = secrets.token_bytes(32)
        private_key = '0x' + private_key_bytes.hex()
        
        # In a real implementation, this would derive the public key using secp256k1
        # For now, we'll create a deterministic placeholder
        public_key_bytes = hashlib.sha256(private_key_bytes).digest()
        public_key = '0x' + public_key_bytes.hex()
        
        return private_key, public_key
    
    @staticmethod
    def hash_message(message: str) -> str:
        """Hash a message using keccak256 (Ethereum standard)"""
        # Using SHA3-256 as a placeholder for Keccak256
        return '0x' + hashlib.sha3_256(message.encode()).hexdigest()
    
    @staticmethod
    def sign_message(message: str, private_key: str) -> str:
        """Sign a message with a private key"""
        # Remove 0x prefix if present
        private_key_clean = private_key[2:] if private_key.startswith('0x') else private_key
        
        # Create HMAC signature as placeholder for ECDSA
        signature = hmac.new(
            bytes.fromhex(private_key_clean),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return '0x' + signature
    
    @staticmethod
    def verify_signature(message: str, signature: str, public_key: str) -> bool:
        """Verify a message signature"""
        # This is a placeholder implementation
        # In production, this would use ECDSA signature verification
        try:
            # For now, just check that signature is valid hex
            signature_clean = signature[2:] if signature.startswith('0x') else signature
            int(signature_clean, 16)
            return len(signature_clean) == 64  # 32 bytes = 64 hex chars
        except ValueError:
            return False
    
    @staticmethod
    def generate_random_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_random_hex(length: int) -> str:
        """Generate cryptographically secure random hex string"""
        return '0x' + secrets.token_hex(length)
    
    @staticmethod
    def derive_address_from_public_key(public_key: str) -> str:
        """Derive Ethereum address from public key"""
        # Remove 0x prefix if present
        public_key_clean = public_key[2:] if public_key.startswith('0x') else public_key
        
        # Take the last 20 bytes of the hash
        address_bytes = hashlib.sha3_256(bytes.fromhex(public_key_clean)).digest()[-20:]
        return '0x' + address_bytes.hex()
    
    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        """Encrypt data using symmetric encryption"""
        # Placeholder using XOR with repeated key
        key_clean = key[2:] if key.startswith('0x') else key
        key_bytes = bytes.fromhex(key_clean)
        data_bytes = data.encode()
        
        # Simple XOR encryption (not secure - just for placeholder)
        encrypted = bytearray()
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return '0x' + encrypted.hex()
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        """Decrypt data using symmetric encryption"""
        # Remove 0x prefix if present
        encrypted_clean = encrypted_data[2:] if encrypted_data.startswith('0x') else encrypted_data
        key_clean = key[2:] if key.startswith('0x') else key
        
        key_bytes = bytes.fromhex(key_clean)
        encrypted_bytes = bytes.fromhex(encrypted_clean)
        
        # Simple XOR decryption (same as encryption for XOR)
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return decrypted.decode()
    
    @staticmethod
    def compute_merkle_root(leaves: List[str]) -> str:
        """Compute Merkle tree root from leaf nodes"""
        if not leaves:
            return '0x' + '0' * 64
        
        # Convert all leaves to bytes
        current_level = []
        for leaf in leaves:
            leaf_clean = leaf[2:] if leaf.startswith('0x') else leaf
            current_level.append(bytes.fromhex(leaf_clean))
        
        # Build tree level by level
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # Odd number of elements, promote last one
                    combined = current_level[i]
                
                next_level.append(hashlib.sha256(combined).digest())
            
            current_level = next_level
        
        return '0x' + current_level[0].hex()

__all__ = [
    'errors',
    'validation', 
    'formatting',
    'crypto',
    'A2AError',
    'ErrorCode'
]
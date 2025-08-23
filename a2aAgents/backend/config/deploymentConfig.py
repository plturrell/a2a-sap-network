#!/usr/bin/env python3
"""
Production Configuration Management for A2A Network Integration
Manages environment-specific settings, security, and operational configurations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import subprocess
from urllib.parse import urlparse
from datetime import datetime


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    private_key_encrypted: bool = True
    use_hardware_wallet: bool = False
    require_https: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    allowed_origins: List[str] = None
    enable_cors: bool = False
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

@dataclass
class NetworkConfig:
    """Network configuration for different environments"""
    name: str
    rpc_url: str
    ws_url: Optional[str] = None
    chain_id: int = 1
    block_confirmation_count: int = 1
    gas_price_strategy: str = "medium"  # slow, medium, fast, auto
    max_gas_price_gwei: float = 100.0
    timeout_seconds: int = 30

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    enable_health_checks: bool = True
    health_check_interval: int = 30

@dataclass
class ContractConfig:
    """Smart contract configuration"""
    agent_registry_address: Optional[str] = None
    message_router_address: Optional[str] = None
    ord_registry_address: Optional[str] = None
    deployment_block: Optional[int] = None
    enable_upgrades: bool = False
    proxy_admin_address: Optional[str] = None

class ProductionConfigManager:
    """
    Manages production-ready configuration for A2A Network integration
    """
    
    def __init__(self, environment: Union[Environment, str] = None, config_dir: str = None):
        # Set environment
        if isinstance(environment, str):
            environment = Environment(environment)
        self.environment = environment or self._detect_environment()
        
        # Set configuration directory
        self.config_dir = Path(config_dir or Path(__file__).parent)
        self.config_dir.mkdir(exist_ok=True)
        
        # Load configuration with production-safe defaults
        self.security = SecurityConfig()
        if self.environment == Environment.PRODUCTION:
            self.network = NetworkConfig("mainnet", "")
        else:
            self.network = NetworkConfig("localhost", "os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))")
        self.monitoring = MonitoringConfig()
        self.contracts = ContractConfig()
        
        self._load_configuration()
        
        logger.info(f"Production config manager initialized for {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Auto-detect environment from various sources"""
        # Check environment variable
        env_var = os.getenv("A2A_ENVIRONMENT", "").lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                pass
        
        # Check if running in container
        if os.path.exists("/.dockerenv"):
            return Environment.PRODUCTION if os.getenv("PROD") else Environment.STAGING
        
        # Check if development indicators exist
        dev_indicators = [
            "localhost" in os.getenv("A2A_RPC_URL", ""),
            os.getenv("DEBUG") == "1",
            Path("foundry.toml").exists(),
            Path("package.json").exists()
        ]
        
        if any(dev_indicators):
            return Environment.DEVELOPMENT
        
        # Default to production for safety
        return Environment.PRODUCTION
    
    def _load_configuration(self):
        """Load configuration from multiple sources"""
        # Load from environment-specific files
        config_files = [
            self.config_dir / f"config.{self.environment.value}.json",
            self.config_dir / "config.json",
            self.config_dir / ".env.production" if self.environment == Environment.PRODUCTION else None
        ]
        
        for config_file in config_files:
            if config_file and config_file.exists():
                self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_environment()
        
        # Apply environment-specific defaults
        self._apply_environment_defaults()
    
    def _load_from_file(self, config_file: Path):
        """Load configuration from JSON file"""
        try:
            if config_file.suffix == ".json":
                with open(config_file) as f:
                    config_data = json.load(f)
                
                # Update configuration objects
                if "security" in config_data:
                    self._update_dataclass(self.security, config_data["security"])
                if "network" in config_data:
                    self._update_dataclass(self.network, config_data["network"])
                if "monitoring" in config_data:
                    self._update_dataclass(self.monitoring, config_data["monitoring"])
                if "contracts" in config_data:
                    self._update_dataclass(self.contracts, config_data["contracts"])
                
                logger.info(f"Loaded configuration from {config_file}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Security
            "A2A_PRIVATE_KEY_ENCRYPTED": ("security", "private_key_encrypted", bool),
            "A2A_USE_HARDWARE_WALLET": ("security", "use_hardware_wallet", bool),
            "A2A_REQUIRE_HTTPS": ("security", "require_https", bool),
            "A2A_ENABLE_RATE_LIMITING": ("security", "enable_rate_limiting", bool),
            "A2A_MAX_REQUESTS_PER_MINUTE": ("security", "max_requests_per_minute", int),
            "A2A_ALLOWED_ORIGINS": ("security", "allowed_origins", lambda x: x.split(",")),
            
            # Network
            "A2A_NETWORK": ("network", "name", str),
            "A2A_RPC_URL": ("network", "rpc_url", str),
            "A2A_WS_URL": ("network", "ws_url", str),
            "A2A_CHAIN_ID": ("network", "chain_id", int),
            "A2A_BLOCK_CONFIRMATIONS": ("network", "block_confirmation_count", int),
            "A2A_GAS_STRATEGY": ("network", "gas_price_strategy", str),
            "A2A_MAX_GAS_PRICE": ("network", "max_gas_price_gwei", float),
            
            # Monitoring
            "A2A_ENABLE_METRICS": ("monitoring", "enable_metrics", bool),
            "A2A_METRICS_PORT": ("monitoring", "metrics_port", int),
            "A2A_LOG_LEVEL": ("monitoring", "log_level", str),
            "A2A_LOG_FORMAT": ("monitoring", "log_format", str),
            "A2A_ENABLE_TRACING": ("monitoring", "enable_tracing", bool),
            "A2A_JAEGER_ENDPOINT": ("monitoring", "jaeger_endpoint", str),
            
            # Contracts
            "A2A_AGENT_REGISTRY_ADDRESS": ("contracts", "agent_registry_address", str),
            "A2A_MESSAGE_ROUTER_ADDRESS": ("contracts", "message_router_address", str),
            "A2A_ORD_REGISTRY_ADDRESS": ("contracts", "ord_registry_address", str),
            "A2A_DEPLOYMENT_BLOCK": ("contracts", "deployment_block", int),
            "A2A_ENABLE_UPGRADES": ("contracts", "enable_upgrades", bool),
            "A2A_PROXY_ADMIN_ADDRESS": ("contracts", "proxy_admin_address", str),
        }
        
        for env_var, (section, field, type_converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_converter == bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    elif callable(type_converter):
                        value = type_converter(value)
                    
                    # Get the appropriate configuration object
                    config_obj = getattr(self, section)
                    setattr(config_obj, field, value)
                    
                except Exception as e:
                    logger.error(f"Failed to parse environment variable {env_var}: {e}")
    
    def _apply_environment_defaults(self):
        """Apply environment-specific default configurations"""
        if self.environment == Environment.DEVELOPMENT:
            self.security.require_https = False
            self.security.enable_rate_limiting = False
            self.monitoring.log_level = "DEBUG"
            self.network.block_confirmation_count = 1
            
        elif self.environment == Environment.TESTING:
            self.security.require_https = False
            self.monitoring.log_level = "INFO"
            self.network.block_confirmation_count = 1
            
        elif self.environment == Environment.STAGING:
            self.security.require_https = True
            self.monitoring.log_level = "INFO"
            self.network.block_confirmation_count = 3
            
        elif self.environment == Environment.PRODUCTION:
            self.security.require_https = True
            self.security.enable_rate_limiting = True
            self.monitoring.log_level = "WARNING"
            self.network.block_confirmation_count = 12
            self.monitoring.enable_tracing = True
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the entire configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "environment": self.environment.value
        }
        
        # Validate network configuration
        network_validation = self._validate_network_config()
        if not network_validation["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(network_validation["errors"])
        validation_result["warnings"].extend(network_validation["warnings"])
        
        # Validate security configuration
        security_validation = self._validate_security_config()
        if not security_validation["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(security_validation["errors"])
        validation_result["warnings"].extend(security_validation["warnings"])
        
        # Validate contract configuration
        contract_validation = self._validate_contract_config()
        if not contract_validation["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(contract_validation["errors"])
        validation_result["warnings"].extend(contract_validation["warnings"])
        
        return validation_result
    
    def _validate_network_config(self) -> Dict[str, Any]:
        """Validate network configuration"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check RPC URL
        if not self.network.rpc_url:
            result["errors"].append("Network RPC URL is required")
            result["valid"] = False
        else:
            # Validate URL format
            try:
                parsed = urlparse(self.network.rpc_url)
                if not parsed.scheme or not parsed.netloc:
                    result["errors"].append(f"Invalid RPC URL format: {self.network.rpc_url}")
                    result["valid"] = False
                
                # Check HTTPS requirement
                if self.security.require_https and parsed.scheme != "https":
                    if self.environment == Environment.PRODUCTION:
                        result["errors"].append("HTTPS required for production environment")
                        result["valid"] = False
                    else:
                        result["warnings"].append("HTTPS recommended for secure connections")
                
                # Test connectivity
                if not self._test_rpc_connectivity():
                    result["warnings"].append("Unable to connect to RPC endpoint")
                    
            except Exception as e:
                result["errors"].append(f"RPC URL validation error: {e}")
                result["valid"] = False
        
        # Validate chain ID
        if self.network.chain_id <= 0:
            result["errors"].append("Chain ID must be positive")
            result["valid"] = False
        
        return result
    
    def _validate_security_config(self) -> Dict[str, Any]:
        """Validate security configuration"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check private key configuration
        private_key_env = os.getenv("A2A_AGENT_PRIVATE_KEY")
        if not private_key_env:
            if self.environment == Environment.PRODUCTION:
                result["errors"].append("Private key must be configured for production")
                result["valid"] = False
            else:
                result["warnings"].append("No private key configured - agent will create new identity")
        else:
            # Check for template private keys
            template_keys = [
                "0x0000000000000000000000000000000000000000000000000000000000000000",
                "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "REPLACE_WITH_ENCRYPTED_PRIVATE_KEY"
            ]
            
            if private_key_env in template_keys:
                if self.environment == Environment.PRODUCTION:
                    result["errors"].append("Template private key detected - replace with actual secure key")
                    result["valid"] = False
                else:
                    result["warnings"].append("Template private key detected - replace with actual key")
            
            # Validate private key format (if not template)
            elif private_key_env not in template_keys:
                if not private_key_env.startswith("0x") or len(private_key_env) != 66:
                    result["errors"].append("Invalid private key format - must be 64 hex characters with 0x prefix")
                    result["valid"] = False
        
        # Validate rate limiting settings
        if self.security.enable_rate_limiting:
            if self.security.max_requests_per_minute <= 0:
                result["errors"].append("Rate limit must be positive when enabled")
                result["valid"] = False
        
        # Check production security requirements
        if self.environment == Environment.PRODUCTION:
            if not self.security.require_https:
                result["errors"].append("HTTPS is required in production")
                result["valid"] = False
            
            if not self.security.private_key_encrypted:
                result["warnings"].append("Consider encrypting private key for production")
        
        return result
    
    def _validate_contract_config(self) -> Dict[str, Any]:
        """Validate contract configuration"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        required_contracts = ["agent_registry_address", "message_router_address"]
        
        for contract_field in required_contracts:
            address = getattr(self.contracts, contract_field)
            if not address:
                if self.environment in [Environment.STAGING, Environment.PRODUCTION]:
                    result["errors"].append(f"Contract address required for {contract_field}")
                    result["valid"] = False
                else:
                    result["warnings"].append(f"No {contract_field} configured")
            elif not self._is_valid_address(address):
                result["errors"].append(f"Invalid contract address for {contract_field}: {address}")
                result["valid"] = False
        
        return result
    
    def _test_rpc_connectivity(self) -> bool:
        """Test RPC endpoint connectivity"""
        try:
            parsed = urlparse(self.network.rpc_url)
            if parsed.hostname and parsed.port:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((parsed.hostname, parsed.port))
                sock.close()
                return result == 0
        except Exception:
            pass
        return False
    
    def _is_valid_address(self, address: str) -> bool:
        """Validate Ethereum address format"""
        if not address:
            return False
        
        # Basic format check
        if not address.startswith("0x") or len(address) != 42:
            return False
        
        # Check for template placeholders
        template_indicators = [
            "0x0000000000000000000000000000000000000000",  # Zero address
            "0x1234567890123456789012345678901234567890",  # Template address
            "0x2345678901234567890123456789012345678901",
            "0x3456789012345678901234567890123456789012",
            "0x4567890123456789012345678901234567890123",
            "0x5678901234567890123456789012345678901234",
            "0x6789012345678901234567890123456789012345",
            "0x7890123456789012345678901234567890123456",
            "0x8901234567890123456789012345678901234567"
        ]
        
        if address in template_indicators:
            return False
        
        # Check hex format
        try:
            int(address[2:], 16)
            return True
        except ValueError:
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information"""
        return {
            "environment": self.environment.value,
            "config_dir": str(self.config_dir),
            "security": asdict(self.security),
            "network": asdict(self.network),
            "monitoring": asdict(self.monitoring),
            "contracts": asdict(self.contracts),
            "validation": self.validate_configuration()
        }
    
    def save_configuration(self, filename: str = None) -> bool:
        """Save current configuration to file"""
        try:
            if not filename:
                filename = f"config.{self.environment.value}.json"
            
            config_file = self.config_dir / filename
            
            config_data = {
                "environment": self.environment.value,
                "security": asdict(self.security),
                "network": asdict(self.network),
                "monitoring": asdict(self.monitoring),
                "contracts": asdict(self.contracts)
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def create_environment_template(self, environment: Environment) -> str:
        """Create environment variable template"""
        template_lines = [
            f"# A2A Network Configuration - {environment.value.upper()}",
            f"# Generated on {datetime.now().isoformat()}",
            "",
            "# Environment",
            f"A2A_ENVIRONMENT={environment.value}",
            "",
            "# Security",
            f"A2A_PRIVATE_KEY_ENCRYPTED={'true' if environment == Environment.PRODUCTION else 'false'}",
            f"A2A_REQUIRE_HTTPS={'true' if environment in [Environment.STAGING, Environment.PRODUCTION] else 'false'}",
            f"A2A_ENABLE_RATE_LIMITING={'true' if environment != Environment.DEVELOPMENT else 'false'}",
            "A2A_MAX_REQUESTS_PER_MINUTE=100",
            "",
            "# Network",
            f"A2A_NETWORK={environment.value}",
            "A2A_RPC_URL=https://your-rpc-endpoint.com",
            "A2A_WS_URL=wss://your-ws-endpoint.com",
            "A2A_CHAIN_ID=1",
            f"A2A_BLOCK_CONFIRMATIONS={12 if environment == Environment.PRODUCTION else 3}",
            "A2A_GAS_STRATEGY=medium",
            "A2A_MAX_GAS_PRICE=100.0",
            "",
            "# Monitoring",
            f"A2A_ENABLE_METRICS={'true' if environment != Environment.DEVELOPMENT else 'false'}",
            "A2A_METRICS_PORT=9090",
            f"A2A_LOG_LEVEL={'WARNING' if environment == Environment.PRODUCTION else 'INFO'}",
            "A2A_LOG_FORMAT=json",
            f"A2A_ENABLE_TRACING={'true' if environment == Environment.PRODUCTION else 'false'}",
            "",
            "# Contracts (Update with actual deployed addresses)",
            "A2A_AGENT_REGISTRY_ADDRESS=0x...",
            "A2A_MESSAGE_ROUTER_ADDRESS=0x...",
            "A2A_ORD_REGISTRY_ADDRESS=0x...",
            "A2A_DEPLOYMENT_BLOCK=0",
            f"A2A_ENABLE_UPGRADES={'true' if environment != Environment.PRODUCTION else 'false'}",
            "",
            "# Private Keys (Keep secure!)",
            "A2A_AGENT_PRIVATE_KEY=0x..."
        ]
        
        return "\n".join(template_lines)

# Global production config manager
_prod_config_manager: Optional[ProductionConfigManager] = None

def get_production_config(environment: Union[Environment, str] = None) -> ProductionConfigManager:
    """Get or create global production configuration manager"""
    global _prod_config_manager
    if _prod_config_manager is None:
        _prod_config_manager = ProductionConfigManager(environment)
    return _prod_config_manager

def validate_production_setup() -> bool:
    """Validate that production setup is correctly configured"""
    config = get_production_config()
    validation = config.validate_configuration()
    
    if not validation["valid"]:
        logger.error("Production configuration validation failed:")
        for error in validation["errors"]:
            logger.error(f"  - {error}")
    
    for warning in validation["warnings"]:
        logger.warning(f"Configuration warning: {warning}")
    
    return validation["valid"]
"""
A2A Environment Configuration
Centralized configuration for development and production modes
"""

import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EnvironmentMode(Enum):
    """Environment modes for A2A system"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class A2AConfig:
    """A2A system configuration"""
    mode: EnvironmentMode
    service_url: str
    service_host: str
    base_url: str
    private_key: str
    rpc_url: str
    contract_addresses: Dict[str, str]
    agent_ports: Dict[str, int]
    debug: bool = False
    mock_blockchain: bool = False
    mock_storage: bool = False

class EnvironmentConfig:
    """Manages environment configuration for A2A system"""

    # Default values for different environments
    DEFAULTS = {
        EnvironmentMode.DEVELOPMENT: {
            "A2A_SERVICE_URL": "http://localhost:4004",
            "A2A_SERVICE_HOST": "localhost",
            "A2A_BASE_URL": "http://localhost:4004",
            "A2A_PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "A2A_RPC_URL": "http://localhost:8545",
            "A2A_DEBUG": "true",
            "A2A_MOCK_BLOCKCHAIN": "true",
            "A2A_MOCK_STORAGE": "true"
        },
        EnvironmentMode.TEST: {
            "A2A_SERVICE_URL": "http://localhost:4004",
            "A2A_SERVICE_HOST": "localhost",
            "A2A_BASE_URL": "http://localhost:4004",
            "A2A_PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "A2A_RPC_URL": "http://localhost:8545",
            "A2A_DEBUG": "true",
            "A2A_MOCK_BLOCKCHAIN": "true",
            "A2A_MOCK_STORAGE": "true"
        },
        EnvironmentMode.PRODUCTION: {
            # No defaults for production - all must be explicitly set
        }
    }

    # Agent port mappings
    AGENT_PORTS = {
        "agent0_data_product": 8000,
        "agent1_standardization": 8001,
        "agent2_ai_preparation": 8002,
        "agent3_vector_processing": 8003,
        "agent4_calc_validation": 8004,
        "agent5_qa_validation": 8005,
        "agent6_quality_control": 8006,
        "agent7_agent_manager": 8007,
        "agent8_data_manager": 8008,
        "agent9_reasoning": 8009,
        "agent10_calculation": 8010,
        "agent11_sql": 8011,
        "agent12_catalog_manager": 8012,
        "agent13_agent_builder": 8013,
        "agent14_embedding_finetuner": 8014,
        "agent15_orchestrator": 8015
    }

    # Contract addresses for development
    DEV_CONTRACT_ADDRESSES = {
        "AgentRegistry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "MessageBus": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
        "TrustManager": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
        "StorageManager": "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9"
    }

    @classmethod
    def get_mode(cls) -> EnvironmentMode:
        """Get current environment mode"""
        mode_str = os.getenv("A2A_MODE", "").lower()

        # Check for explicit mode setting
        if mode_str in ["dev", "development"]:
            return EnvironmentMode.DEVELOPMENT
        elif mode_str in ["prod", "production"]:
            return EnvironmentMode.PRODUCTION
        elif mode_str in ["test", "testing"]:
            return EnvironmentMode.TEST

        # Check for dev mode flag
        if os.getenv("A2A_DEV_MODE", "false").lower() == "true":
            return EnvironmentMode.DEVELOPMENT

        # Default to production
        return EnvironmentMode.PRODUCTION

    @classmethod
    def load_config(cls) -> A2AConfig:
        """Load configuration based on environment"""
        mode = cls.get_mode()

        # Apply defaults for non-production modes
        if mode != EnvironmentMode.PRODUCTION:
            defaults = cls.DEFAULTS.get(mode, {})
            for key, value in defaults.items():
                if key not in os.environ:
                    os.environ[key] = value
                    logger.info(f"{mode.value} mode: Using default for {key}")

        # Validate required variables
        required_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            if mode == EnvironmentMode.PRODUCTION:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            else:
                raise ValueError(f"Failed to set defaults for: {missing_vars}")

        # Load contract addresses
        contract_addresses = cls.DEV_CONTRACT_ADDRESSES if mode != EnvironmentMode.PRODUCTION else {}

        # Override with environment variables if set
        for contract_name in ["AgentRegistry", "MessageBus", "TrustManager", "StorageManager"]:
            env_var = f"A2A_CONTRACT_{contract_name.upper()}"
            if os.getenv(env_var):
                contract_addresses[contract_name] = os.getenv(env_var)

        # Create configuration
        config = A2AConfig(
            mode=mode,
            service_url=os.getenv("A2A_SERVICE_URL"),
            service_host=os.getenv("A2A_SERVICE_HOST"),
            base_url=os.getenv("A2A_BASE_URL"),
            private_key=os.getenv("A2A_PRIVATE_KEY", ""),
            rpc_url=os.getenv("A2A_RPC_URL", "http://localhost:8545"),
            contract_addresses=contract_addresses,
            agent_ports=cls.AGENT_PORTS,
            debug=os.getenv("A2A_DEBUG", "false").lower() == "true",
            mock_blockchain=os.getenv("A2A_MOCK_BLOCKCHAIN", "false").lower() == "true",
            mock_storage=os.getenv("A2A_MOCK_STORAGE", "false").lower() == "true"
        )

        logger.info(f"Loaded A2A configuration for {mode.value} mode")
        if config.debug:
            logger.debug(f"Configuration: {config}")

        return config

    @classmethod
    def set_development_mode(cls):
        """Convenience method to set development mode"""
        os.environ["A2A_MODE"] = "development"
        os.environ["A2A_DEV_MODE"] = "true"

    @classmethod
    def set_production_mode(cls):
        """Convenience method to set production mode"""
        os.environ["A2A_MODE"] = "production"
        os.environ["A2A_DEV_MODE"] = "false"

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.get_mode() == EnvironmentMode.DEVELOPMENT

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return cls.get_mode() == EnvironmentMode.PRODUCTION

    @classmethod
    def is_test(cls) -> bool:
        """Check if running in test mode"""
        return cls.get_mode() == EnvironmentMode.TEST

# Global configuration instance
_config: Optional[A2AConfig] = None

def get_config() -> A2AConfig:
    """Get or create global configuration"""
    global _config
    if _config is None:
        _config = EnvironmentConfig.load_config()
    return _config

def reload_config() -> A2AConfig:
    """Reload configuration from environment"""
    global _config
    _config = EnvironmentConfig.load_config()
    return _config

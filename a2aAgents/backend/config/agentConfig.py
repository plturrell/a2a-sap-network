"""
Centralized configuration for A2A agents.
Replaces hardcoded values with environment-based configuration.
"""
import os
import logging
from typing import Dict, Optional
from pathlib import Path


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class AgentConfig:
    """Centralized configuration for all A2A agents."""
    
    def __init__(self):
        # Base URLs
        self.base_url = os.getenv("A2A_BASE_URL")
        self.agent_network_url = os.getenv("AGENT_NETWORK_URL", f"{self.base_url}")
        self.data_manager_url = os.getenv("DATA_MANAGER_URL", f"{self.base_url.replace('8000', '8001')}")
        self.catalog_manager_url = os.getenv("CATALOG_MANAGER_URL", f"{self.base_url.replace('8000', '8002')}")
        self.agent_manager_url = os.getenv("AGENT_MANAGER_URL", f"{self.base_url.replace('8000', '8003')}")
        self.qa_validation_url = os.getenv("QA_VALIDATION_URL", f"{self.base_url.replace('8000', '8007')}")
        
        # Storage paths - Use proper storage configuration
        from app.a2a.config.storageConfig import get_storage_config
        storage_config = get_storage_config()
        self.storage_base_path = storage_config.base_storage_path
        self.agent_manager_storage = self.storage_base_path / "agent_manager_state"
        self.data_product_storage = self.storage_base_path / "data_product_state"
        
        # Blockchain configuration
        self.blockchain_network = os.getenv("BLOCKCHAIN_NETWORK", "mainnet")
        self.blockchain_rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "https://mainnet.infura.io/v3/YOUR-PROJECT-ID")
        self.contract_addresses = self._load_contract_addresses()
        
        # Performance settings
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "100"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Security settings
        self.enable_ssl = os.getenv("ENABLE_SSL", "true").lower() == "true"
        self.cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        
        # HANA configuration
        self.hana_host = os.getenv("HANA_HOST", "localhost")
        self.hana_port = os.getenv("HANA_PORT", "30015")
        self.hana_user = os.getenv("HANA_USER", "SYSTEM")
        self.hana_password = os.getenv("HANA_PASSWORD", "")
        self.hana_database = os.getenv("HANA_DATABASE", "A2A")
        self.hana_schema = os.getenv("HANA_SCHEMA", "A2A_VECTORS")
        
        # Create storage directories if they don't exist
        self._ensure_directories()
    
    def _load_contract_addresses(self) -> Dict[str, str]:
        """Load contract addresses from environment variables."""
        return {
            "agent_manager": os.getenv("CONTRACT_AGENT_MANAGER", ""),
            "data_product": os.getenv("CONTRACT_DATA_PRODUCT", ""),
            "data_standardization": os.getenv("CONTRACT_DATA_STANDARDIZATION", ""),
            "ai_preparation": os.getenv("CONTRACT_AI_PREPARATION", ""),
            "vector_processing": os.getenv("CONTRACT_VECTOR_PROCESSING", ""),
            "calc_validation": os.getenv("CONTRACT_CALC_VALIDATION", ""),
            "qa_validation": os.getenv("CONTRACT_QA_VALIDATION", ""),
            "data_manager": os.getenv("CONTRACT_DATA_MANAGER", ""),
            "catalog_manager": os.getenv("CONTRACT_CATALOG_MANAGER", ""),
            "agent_builder": os.getenv("CONTRACT_AGENT_BUILDER", ""),
            "enhanced_calculation": os.getenv("CONTRACT_ENHANCED_CALCULATION", ""),
            "reasoning": os.getenv("CONTRACT_REASONING", ""),
            "sql_agent": os.getenv("CONTRACT_SQL_AGENT", ""),
            "developer_portal": os.getenv("CONTRACT_DEVELOPER_PORTAL", ""),
            "trust_registry": os.getenv("CONTRACT_TRUST_REGISTRY", ""),
            "reputation_exchange": os.getenv("CONTRACT_REPUTATION_EXCHANGE", ""),
        }
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        self.storage_base_path.mkdir(parents=True, exist_ok=True)
        self.agent_manager_storage.mkdir(parents=True, exist_ok=True)
        self.data_product_storage.mkdir(parents=True, exist_ok=True)
    
    def get_agent_url(self, agent_type: str) -> str:
        """Get URL for a specific agent type."""
        url_map = {
            "agent_network": self.agent_network_url,
            "data_manager": self.data_manager_url,
            "catalog_manager": self.catalog_manager_url,
            "agent_manager": self.agent_manager_url,
            "qa_validation": self.qa_validation_url,
        }
        return url_map.get(agent_type, self.base_url)
    
    def get_contract_address(self, contract_name: str) -> Optional[str]:
        """Get contract address for a specific contract."""
        address = self.contract_addresses.get(contract_name)
        if not address:
            raise ValueError(f"Contract address for '{contract_name}' not configured. "
                           f"Set the environment variable CONTRACT_{contract_name.upper()}")
        return address
    
    def validate_production_config(self) -> bool:
        """Validate that all required production configurations are set."""
        errors = []
        
        # Check URLs
        if "localhost" in self.base_url and self.blockchain_network == "mainnet":
            errors.append("Production deployment cannot use localhost URLs")
        
        # Check contract addresses
        for name, address in self.contract_addresses.items():
            if not address or address == "0x0000000000000000000000000000000000000000":
                errors.append(f"Contract '{name}' has no valid address configured")
        
        # Check RPC URL
        if "YOUR-PROJECT-ID" in self.blockchain_rpc_url:
            errors.append("Blockchain RPC URL not properly configured")
        
        # Check storage paths
        if str(self.storage_base_path).startswith("/tmp") and self.blockchain_network == "mainnet":
            errors.append("Production deployment should not use /tmp for storage")
        
        if errors:
            logger.error("Configuration validation errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True


# Global configuration instance
config = AgentConfig()
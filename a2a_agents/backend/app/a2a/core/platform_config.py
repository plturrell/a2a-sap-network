"""
Platform Configuration with Real Endpoints
Contains actual API endpoints and configuration for supported platforms
"""

from typing import Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class PlatformEndpoints:
    """Platform-specific endpoint configuration"""
    base_url: str
    api_version: str
    endpoints: Dict[str, str]
    auth_endpoints: Dict[str, str]
    documentation: str


# SAP Datasphere Configuration
SAP_DATASPHERE_CONFIG = PlatformEndpoints(
    base_url=os.getenv("SAP_DATASPHERE_URL", "https://api.datasphere.cloud.sap"),
    api_version="v1",
    endpoints={
        "catalog_sync": "/api/v1/catalog/import",
        "health": "/api/v1/system/health",
        "spaces": "/api/v1/spaces",
        "data_marketplace": "/api/v1/marketplace/publish",
        "metadata": "/api/v1/metadata",
        "ord_registry": "/api/v1/ord/documents"
    },
    auth_endpoints={
        "token_url": "https://authentication.datasphere.cloud.sap/oauth/token",
        "authorize_url": "https://authentication.datasphere.cloud.sap/oauth/authorize"
    },
    documentation="https://api.datasphere.cloud.sap/api-docs"
)

# Databricks Unity Catalog Configuration
DATABRICKS_UNITY_CONFIG = PlatformEndpoints(
    base_url=os.getenv("DATABRICKS_HOST", "https://dbc-xxxxx.cloud.databricks.com"),
    api_version="2.1",
    endpoints={
        "catalogs": "/api/2.1/unity-catalog/catalogs",
        "schemas": "/api/2.1/unity-catalog/schemas",
        "tables": "/api/2.1/unity-catalog/tables",
        "volumes": "/api/2.1/unity-catalog/volumes",
        "metastores": "/api/2.1/unity-catalog/metastores",
        "delta_sharing": "/api/2.0/delta-sharing/shares"
    },
    auth_endpoints={
        "token_endpoint": "/api/2.0/token/create",
        "service_principal": "/api/2.0/service-principals"
    },
    documentation="https://docs.databricks.com/api/workspace/introduction"
)

# SAP HANA Cloud Configuration
SAP_HANA_CONFIG = PlatformEndpoints(
    base_url=os.getenv("HANA_ENDPOINT", "https://api.cf.sap.hana.ondemand.com"),
    api_version="v1",
    endpoints={
        "hdi_deploy": "/v1/deploy",
        "containers": "/v1/containers",
        "status": "/v1/status",
        "artifacts": "/v1/artifacts",
        "grants": "/v1/grants",
        "build": "/v1/build"
    },
    auth_endpoints={
        "token_url": "https://login.cf.sap.hana.ondemand.com/oauth/token",
        "uaa_url": "https://uaa.cf.sap.hana.ondemand.com"
    },
    documentation="https://help.sap.com/docs/HANA_CLOUD"
)

# Cloudera Data Platform Atlas Configuration
CLOUDERA_ATLAS_CONFIG = PlatformEndpoints(
    base_url=os.getenv("ATLAS_ENDPOINT", "https://atlas.cloudera.com"),
    api_version="v2",
    endpoints={
        "entities": "/api/atlas/v2/entity",
        "bulk": "/api/atlas/v2/entity/bulk",
        "search": "/api/atlas/v2/search/basic",
        "typedefs": "/api/atlas/v2/types/typedefs",
        "lineage": "/api/atlas/v2/lineage",
        "glossary": "/api/atlas/v2/glossary"
    },
    auth_endpoints={
        "knox_gateway": "/gateway/cdp-proxy/knox/api/v1/token",
        "ldap": "/api/atlas/v2/auth/ldap"
    },
    documentation="https://atlas.apache.org/#/api/v2"
)


class PlatformConfigManager:
    """Manage platform configurations"""
    
    def __init__(self):
        self.configs = {
            "sap_datasphere": SAP_DATASPHERE_CONFIG,
            "databricks": DATABRICKS_UNITY_CONFIG,
            "sap_hana": SAP_HANA_CONFIG,
            "cloudera": CLOUDERA_ATLAS_CONFIG
        }
    
    def get_platform_config(self, platform_type: str) -> Optional[PlatformEndpoints]:
        """Get configuration for platform type"""
        return self.configs.get(platform_type)
    
    def get_endpoint(self, platform_type: str, endpoint_name: str) -> Optional[str]:
        """Get specific endpoint for platform"""
        config = self.configs.get(platform_type)
        if config:
            base = config.base_url.rstrip('/')
            endpoint = config.endpoints.get(endpoint_name, "")
            return f"{base}{endpoint}" if endpoint else None
        return None
    
    def get_auth_config(self, platform_type: str) -> Dict[str, str]:
        """Get authentication configuration for platform"""
        config = self.configs.get(platform_type)
        if config:
            return config.auth_endpoints
        return {}
    
    def validate_platform_config(self, platform_type: str) -> Dict[str, Any]:
        """Validate platform configuration"""
        config = self.configs.get(platform_type)
        if not config:
            return {"valid": False, "error": f"Unknown platform: {platform_type}"}
        
        issues = []
        
        # Check base URL
        if not config.base_url or config.base_url.startswith("https://xxx"):
            issues.append("Base URL not configured")
        
        # Check required environment variables
        env_checks = {
            "sap_datasphere": ["SAP_CLIENT_ID", "SAP_CLIENT_SECRET"],
            "databricks": ["DATABRICKS_TOKEN"],
            "sap_hana": ["HANA_CLIENT_ID", "HANA_CLIENT_SECRET"],
            "cloudera": ["ATLAS_USERNAME", "ATLAS_PASSWORD"]
        }
        
        required_vars = env_checks.get(platform_type, [])
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": config.__dict__ if len(issues) == 0 else None
        }


# Global config manager
_config_manager = None

def get_platform_config_manager() -> PlatformConfigManager:
    """Get global platform config manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = PlatformConfigManager()
    return _config_manager
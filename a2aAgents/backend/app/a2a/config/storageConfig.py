"""
A2A Storage Configuration
Centralized storage paths and configuration management
"""

import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StorageConfig:
    """Centralized storage configuration for A2A system"""

    def __init__(self):
        # Base storage directory from environment or default to user data dir
        self.base_storage_path = self._get_base_storage_path()

        # Ensure base directory exists
        self.base_storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ A2A storage initialized at: {self.base_storage_path}")

    def _get_base_storage_path(self) -> Path:
        """Get base storage path from environment or system default"""

        # Check environment variable first
        env_path = os.getenv("A2A_STORAGE_PATH")
        if env_path:
            return Path(env_path)

        # Use system-appropriate data directory
        if os.name == 'nt':  # Windows
            data_dir = Path(os.getenv("APPDATA", "")) / "A2ANetwork"
        else:  # Unix-like systems
            # Try XDG_DATA_HOME first, fallback to ~/.local/share
            xdg_data = os.getenv("XDG_DATA_HOME")
            if xdg_data:
                data_dir = Path(xdg_data) / "a2a-network"
            else:
                data_dir = Path.home() / ".local" / "share" / "a2a-network"

        return data_dir

    @property
    def trust_storage_path(self) -> Path:
        """Trust system identities and contracts storage"""
        path = self.base_storage_path / "trust"
        path.mkdir(exist_ok=True)
        return path

    @property
    def workspace_path(self) -> Path:
        """Developer workspace storage"""
        path = self.base_storage_path / "workspace"
        path.mkdir(exist_ok=True)
        return path

    @property
    def projects_path(self) -> Path:
        """Developer projects storage"""
        path = self.base_storage_path / "projects"
        path.mkdir(exist_ok=True)
        return path

    @property
    def registry_cache_path(self) -> Path:
        """Registry and service discovery cache"""
        path = self.base_storage_path / "registry"
        path.mkdir(exist_ok=True)
        return path

    @property
    def workflow_storage_path(self) -> Path:
        """Workflow execution persistence"""
        path = self.base_storage_path / "workflows"
        path.mkdir(exist_ok=True)
        return path

    @property
    def logs_path(self) -> Path:
        """Application logs storage"""
        path = self.base_storage_path / "logs"
        path.mkdir(exist_ok=True)
        return path

    @property
    def cache_path(self) -> Path:
        """General cache storage"""
        path = self.base_storage_path / "cache"
        path.mkdir(exist_ok=True)
        return path

    @property
    def reports_path(self) -> Path:
        """System reports and diagnostics"""
        path = self.base_storage_path / "reports"
        path.mkdir(exist_ok=True)
        return path

    def get_agent_storage_path(self, agent_id: str) -> Path:
        """Get agent-specific storage directory"""
        path = self.base_storage_path / "agents" / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_service_storage_path(self, service_name: str) -> Path:
        """Get service-specific storage directory"""
        path = self.base_storage_path / "services" / service_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_temp_data(self, max_age_days: int = 7):
        """Clean up temporary data older than specified days"""
        logger.info(f"Cleaning up temporary data older than {max_age_days} days")
        # Implementation would clean old cache, logs, etc.
        pass

    def get_storage_info(self) -> Dict[str, str]:
        """Get storage configuration information"""
        return {
            "base_path": str(self.base_storage_path),
            "trust_storage": str(self.trust_storage_path),
            "workspace": str(self.workspace_path),
            "registry_cache": str(self.registry_cache_path),
            "workflows": str(self.workflow_storage_path),
            "logs": str(self.logs_path),
            "cache": str(self.cache_path),
            "reports": str(self.reports_path)
        }


# Global storage configuration instance
_storage_config: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """Get singleton storage configuration instance"""
    global _storage_config
    if _storage_config is None:
        _storage_config = StorageConfig()
    return _storage_config


# Convenience functions
def get_trust_storage_path() -> Path:
    """Get trust system storage path"""
    return get_storage_config().trust_storage_path


def get_workspace_path() -> Path:
    """Get workspace storage path"""
    return get_storage_config().workspace_path


def get_logs_path() -> Path:
    """Get logs storage path"""
    return get_storage_config().logs_path


def get_workflow_storage_path() -> Path:
    """Get workflow storage path"""
    return get_storage_config().workflow_storage_path


def get_registry_cache_path() -> Path:
    """Get registry cache path"""
    return get_storage_config().registry_cache_path


def get_agent_storage_path(agent_id: str) -> Path:
    """Get agent-specific storage path"""
    return get_storage_config().get_agent_storage_path(agent_id)


if __name__ == "__main__":
    # Test storage configuration
    config = get_storage_config()
    print("🗄️  A2A Storage Configuration")
    print("=" * 40)

    for name, path in config.get_storage_info().items():
        print(f"📁 {name:<15}: {path}")

    print(f"\n✅ Storage configuration working correctly!")

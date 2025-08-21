#!/usr/bin/env python3
"""
Centralized Configuration Management System for A2A Network

This module provides a unified configuration management system that:
1. Loads configuration from multiple sources (files, environment, remote)
2. Supports environment-specific overrides
3. Provides type-safe configuration access
4. Enables hot-reloading of configuration
5. Integrates with observability for configuration changes
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import BaseSettings, validator
from contextlib import asynccontextmanager
import aiofiles
import etcd3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration sources in order of precedence."""
    ENVIRONMENT = "environment"
    CLI_ARGS = "cli_args"
    CONFIG_FILE = "config_file"
    ETCD = "etcd"
    DEFAULTS = "defaults"


@dataclass
class ConfigMetadata:
    """Metadata for configuration values."""
    source: ConfigSource
    last_updated: str
    version: str = "1.0"
    description: str = ""
    sensitive: bool = False


class A2AConfig(BaseSettings):
    """Main configuration class with validation."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Network
    host: str = "localhost"
    base_port: int = 8000
    
    # Agent Configuration
    agent_timeout: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    
    # Database
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    
    # Blockchain
    blockchain_network: str = "development"
    private_key: Optional[str] = None
    contract_address: Optional[str] = None
    
    # Security
    jwt_secret: str = "dev-secret-key"
    cors_origins: List[str] = [os.getenv("A2A_SERVICE_URL")]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Telemetry
    telemetry_endpoint: Optional[str] = None
    telemetry_enabled: bool = True
    metrics_port: int = 9090
    
    # Agent-specific ports
    agent0_port: int = 8001  # Data Product Agent
    agent1_port: int = 8002  # Standardization Agent
    agent2_port: int = 8003  # AI Preparation Agent
    agent3_port: int = 8004  # Vector Processing Agent
    agent4_port: int = 8009  # Calculation Validation Agent
    agent5_port: int = 8010  # QA Validation Agent
    data_manager_port: int = 8005
    catalog_manager_port: int = 8006
    agent_manager_port: int = 8007
    
    class Config:
        env_prefix = "A2A_"
        case_sensitive = False
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v


class ConfigFileWatcher(FileSystemEventHandler):
    """Watch configuration files for changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager._reload_config())


class ConfigurationManager:
    """Centralized configuration management system."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent
        self.config: A2AConfig = A2AConfig()
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.observers: List[Observer] = []
        self.etcd_client: Optional[etcd3.Etcd3Client] = None
        self.config_cache: Dict[str, Any] = {}
        self.subscribers: List[callable] = []
        
    async def initialize(self, 
                        environment: str = "development",
                        config_file: Optional[str] = None,
                        etcd_host: Optional[str] = None):
        """Initialize the configuration manager."""
        logger.info("Initializing configuration manager...")
        
        # Load configuration in order of precedence
        await self._load_defaults()
        
        if etcd_host:
            await self._setup_etcd(etcd_host)
            await self._load_from_etcd(environment)
        
        if config_file:
            await self._load_from_file(config_file)
        
        await self._load_from_environment()
        
        # Start file watcher for hot-reload
        await self._setup_file_watcher()
        
        logger.info(f"Configuration loaded for environment: {environment}")
    
    async def _load_defaults(self):
        """Load default configuration."""
        self.config = A2AConfig()
        self._update_metadata("defaults", ConfigSource.DEFAULTS, "Default configuration")
    
    async def _load_from_file(self, config_file: str):
        """Load configuration from file."""
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()
                
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
            
            # Update configuration
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self._update_metadata(key, ConfigSource.CONFIG_FILE, f"From {config_file}")
            
            logger.info(f"Configuration loaded from file: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    async def _load_from_environment(self):
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith("A2A_"):
                config_key = key[4:].lower()  # Remove A2A_ prefix
                if hasattr(self.config, config_key):
                    # Type conversion based on field type
                    field_type = type(getattr(self.config, config_key))
                    if field_type == bool:
                        value = value.lower() in ('true', '1', 'yes')
                    elif field_type == int:
                        value = int(value)
                    elif field_type == list:
                        value = value.split(',')
                    
                    setattr(self.config, config_key, value)
                    self._update_metadata(config_key, ConfigSource.ENVIRONMENT, 
                                        f"From environment variable {key}")
    
    async def _setup_etcd(self, etcd_host: str):
        """Setup etcd client for distributed configuration."""
        try:
            self.etcd_client = etcd3.client(host=etcd_host)
            logger.info(f"Connected to etcd at {etcd_host}")
        except Exception as e:
            logger.error(f"Failed to connect to etcd: {e}")
            self.etcd_client = None
    
    async def _load_from_etcd(self, environment: str):
        """Load configuration from etcd."""
        if not self.etcd_client:
            return
        
        try:
            # Load configuration from etcd with environment prefix
            prefix = f"/a2a/{environment}/"
            for value, metadata in self.etcd_client.get_prefix(prefix):
                if value:
                    key = metadata.key.decode('utf-8').replace(prefix, '')
                    config_key = key.replace('-', '_')
                    
                    if hasattr(self.config, config_key):
                        # Parse JSON values
                        try:
                            parsed_value = json.loads(value.decode('utf-8'))
                        except json.JSONDecodeError:
                            parsed_value = value.decode('utf-8')
                        
                        setattr(self.config, config_key, parsed_value)
                        self._update_metadata(config_key, ConfigSource.ETCD, 
                                            f"From etcd key {metadata.key.decode()}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration from etcd: {e}")
    
    async def _setup_file_watcher(self):
        """Setup file system watcher for configuration hot-reload."""
        try:
            watcher = ConfigFileWatcher(self)
            observer = Observer()
            observer.schedule(watcher, str(self.base_path), recursive=True)
            observer.start()
            self.observers.append(observer)
            logger.info("Configuration file watcher started")
        except Exception as e:
            logger.error(f"Failed to setup file watcher: {e}")
    
    def _update_metadata(self, key: str, source: ConfigSource, description: str):
        """Update configuration metadata."""
        self.metadata[key] = ConfigMetadata(
            source=source,
            last_updated=str(asyncio.get_event_loop().time()),
            description=description
        )
    
    async def _reload_config(self):
        """Reload configuration from all sources."""
        logger.info("Reloading configuration...")
        await self.initialize()
        await self._notify_subscribers()
    
    async def _notify_subscribers(self):
        """Notify all subscribers of configuration changes."""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.config)
                else:
                    callback(self.config)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: callable):
        """Subscribe to configuration changes."""
        self.subscribers.append(callback)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any, persist: bool = False):
        """Set configuration value."""
        setattr(self.config, key, value)
        self._update_metadata(key, ConfigSource.CLI_ARGS, "Set programmatically")
        
        if persist and self.etcd_client:
            # Persist to etcd
            etcd_key = f"/a2a/{self.config.environment}/{key.replace('_', '-')}"
            self.etcd_client.put(etcd_key, json.dumps(value))
    
    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """Get metadata for a configuration key."""
        return self.metadata.get(key)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configuration with metadata."""
        summary = {}
        for key in dir(self.config):
            if not key.startswith('_') and not key == 'Config':
                value = getattr(self.config, key)
                metadata = self.get_metadata(key)
                
                # Mask sensitive values
                if metadata and metadata.sensitive:
                    display_value = "***MASKED***"
                else:
                    display_value = value
                
                summary[key] = {
                    "value": display_value,
                    "source": metadata.source.value if metadata else "unknown",
                    "description": metadata.description if metadata else ""
                }
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform configuration health check."""
        health = {
            "status": "healthy",
            "environment": self.config.environment,
            "sources": list(set(m.source.value for m in self.metadata.values())),
            "etcd_connected": self.etcd_client is not None,
            "watchers_active": len(self.observers)
        }
        
        return health
    
    async def shutdown(self):
        """Shutdown configuration manager."""
        logger.info("Shutting down configuration manager...")
        
        # Stop file watchers
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        # Close etcd connection
        if self.etcd_client:
            self.etcd_client.close()
        
        logger.info("Configuration manager shutdown complete")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


async def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
        await _config_manager.initialize()
    return _config_manager


@asynccontextmanager
async def config_manager(environment: str = "development", 
                        config_file: Optional[str] = None,
                        etcd_host: Optional[str] = None):
    """Context manager for configuration management."""
    manager = ConfigurationManager()
    await manager.initialize(environment, config_file, etcd_host)
    try:
        yield manager
    finally:
        await manager.shutdown()
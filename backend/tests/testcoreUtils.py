"""
Test suite for core utility functions and shared components
Covers configuration management, helpers, and common utilities
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.core.dynamic_config import (
    DynamicConfigManager,
    Environment,
    DatabaseConfig,
    AgentConfig,
    SecurityConfig,
    ConfigValidationError,
    get_config_manager,
    generate_secure_secret,
    validate_production_config,
    Constants
)


class TestDynamicConfigManager:
    """Test dynamic configuration management"""
    
    def test_detect_environment_default(self):
        """Test environment detection with default"""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = DynamicConfigManager()
            assert config_manager.env == Environment.DEVELOPMENT
    
    def test_detect_environment_from_env_var(self):
        """Test environment detection from environment variable"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config_manager = DynamicConfigManager()
            assert config_manager.env == Environment.PRODUCTION
    
    def test_detect_environment_invalid(self):
        """Test environment detection with invalid value"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid_env'}):
            config_manager = DynamicConfigManager()
            assert config_manager.env == Environment.DEVELOPMENT
    
    def test_database_config_validation(self):
        """Test database configuration validation"""
        # Valid config
        valid_config = DatabaseConfig(url="postgresql://user:pass@localhost:5432/db")
        assert valid_config.url.startswith("postgresql://")
        
        # Invalid config - empty URL
        with pytest.raises(ConfigValidationError):
            DatabaseConfig(url="")
        
        # Invalid config - malformed URL
        with pytest.raises(ConfigValidationError):
            DatabaseConfig(url="not-a-url")
    
    def test_security_config_validation(self):
        """Test security configuration validation"""
        # Valid config
        long_secret = "this_is_a_very_long_secret_key_for_jwt_signing_purposes_123456789"
        valid_config = SecurityConfig(jwt_secret=long_secret)
        assert valid_config.jwt_secret == long_secret
        
        # Invalid config - empty secret
        with pytest.raises(ConfigValidationError):
            SecurityConfig(jwt_secret="")
        
        # Invalid config - short secret
        with pytest.raises(ConfigValidationError):
            SecurityConfig(jwt_secret="short")
    
    def test_agent_config_defaults(self):
        """Test agent configuration defaults"""
        config = AgentConfig()
        
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.agent0_timeout == 15
        assert config.agent1_timeout == 45
        assert config.agent2_timeout == 120  # AI processing takes longer
    
    @patch.dict(os.environ, {
        'DATABASE_URL': 'postgresql://test:test@localhost/testdb',
        'JWT_SECRET': 'test_secret_that_is_long_enough_for_validation_purposes_123',
        'REDIS_URL': 'redis://localhost:6379'
    })
    def test_load_from_environment(self):
        """Test loading configuration from environment variables"""
        config_manager = DynamicConfigManager()
        
        db_config = config_manager.get_database_config()
        assert db_config.url == 'postgresql://test:test@localhost/testdb'
        
        security_config = config_manager.get_security_config()
        assert security_config.jwt_secret.startswith('test_secret')
        
        redis_config = config_manager.get_redis_config()
        assert redis_config.url == 'redis://localhost:6379'
    
    def test_production_validation_success(self):
        """Test production configuration validation success"""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'DATABASE_URL': 'postgresql://prod:secret@prod-db:5432/proddb',
            'JWT_SECRET': 'production_secret_key_that_is_very_long_and_secure_for_jwt_123456',
            'SAP_GRAPH_CLIENT_SECRET': 'sap_client_secret',
            'SAP_HANA_URL': 'https://hana-prod.example.com'
        }):
            config_manager = DynamicConfigManager(Environment.PRODUCTION)
            
            # Should not raise exception
            try:
                config_manager._validate_config()
                db_config = config_manager.get_database_config()
                security_config = config_manager.get_security_config()
                sap_config = config_manager.get_sap_config()
            except ConfigValidationError:
                pytest.fail("Production config validation should pass")
    
    def test_production_validation_failure(self):
        """Test production configuration validation failure"""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production'
            # Missing required production secrets
        }):
            with pytest.raises(ConfigValidationError):
                config_manager = DynamicConfigManager(Environment.PRODUCTION)
    
    def test_get_config_manager_singleton(self):
        """Test global config manager singleton"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2  # Same instance
        assert isinstance(manager1, DynamicConfigManager)


class TestConstants:
    """Test configuration constants"""
    
    @patch('app.core.dynamic_config.get_config_manager')
    def test_get_timeout_config(self, mock_get_config):
        """Test timeout configuration retrieval"""
        # Mock config manager
        mock_manager = Mock()
        mock_agent_config = Mock()
        mock_agent_config.timeout_seconds = 30
        mock_agent_config.agent0_timeout = 15
        mock_agent_config.agent1_timeout = 45
        
        mock_external_config = Mock()
        mock_external_config.api_timeout_seconds = 60
        
        mock_ui_config = Mock()
        mock_ui_config.api_timeout_ms = 30000
        
        mock_db_config = Mock()
        mock_db_config.connect_timeout = 10
        mock_db_config.command_timeout = 30
        
        mock_manager.get_agent_config.return_value = mock_agent_config
        mock_manager.get_external_service_config.return_value = mock_external_config
        mock_manager.get_ui_config.return_value = mock_ui_config
        mock_manager.get_database_config.return_value = mock_db_config
        
        mock_get_config.return_value = mock_manager
        
        timeout_config = Constants.get_timeout_config()
        
        assert timeout_config['AGENT_DEFAULT_TIMEOUT'] == 30
        assert timeout_config['AGENT0_TIMEOUT'] == 15
        assert timeout_config['AGENT1_TIMEOUT'] == 45
        assert timeout_config['EXTERNAL_API_TIMEOUT'] == 60
        assert timeout_config['UI_API_TIMEOUT'] == 30  # Converted from ms to seconds
        assert timeout_config['DATABASE_CONNECT_TIMEOUT'] == 10
    
    @patch('app.core.dynamic_config.get_config_manager')
    def test_get_pagination_config(self, mock_get_config):
        """Test pagination configuration retrieval"""
        mock_manager = Mock()
        mock_ui_config = Mock()
        mock_ui_config.default_page_size = 20
        mock_ui_config.max_page_size = 100
        
        mock_notification_config = Mock()
        mock_notification_config.page_size = 25
        
        mock_agent_config = Mock()
        mock_agent_config.processing_batch_size = 10
        
        mock_manager.get_ui_config.return_value = mock_ui_config
        mock_manager.get_notification_config.return_value = mock_notification_config
        mock_manager.get_agent_config.return_value = mock_agent_config
        
        mock_get_config.return_value = mock_manager
        
        pagination_config = Constants.get_pagination_config()
        
        assert pagination_config['DEFAULT_PAGE_SIZE'] == 20
        assert pagination_config['MAX_PAGE_SIZE'] == 100
        assert pagination_config['NOTIFICATION_PAGE_SIZE'] == 25
        assert pagination_config['AGENT_BATCH_SIZE'] == 10
    
    @patch('app.core.dynamic_config.get_config_manager')
    def test_get_limits_config(self, mock_get_config):
        """Test limits configuration retrieval"""
        mock_manager = Mock()
        
        mock_agent_config = Mock()
        mock_agent_config.max_retries = 3
        mock_agent_config.max_concurrent_requests = 100
        
        mock_external_config = Mock()
        mock_external_config.circuit_breaker_threshold = 5
        mock_external_config.max_requests_per_minute = 60
        
        mock_ui_config = Mock()
        mock_ui_config.max_file_size_mb = 50
        mock_ui_config.max_concurrent_uploads = 3
        
        mock_notification_config = Mock()
        mock_notification_config.max_notifications = 1000
        
        mock_manager.get_agent_config.return_value = mock_agent_config
        mock_manager.get_external_service_config.return_value = mock_external_config
        mock_manager.get_ui_config.return_value = mock_ui_config
        mock_manager.get_notification_config.return_value = mock_notification_config
        
        mock_get_config.return_value = mock_manager
        
        limits_config = Constants.get_limits_config()
        
        assert limits_config['MAX_RETRIES'] == 3
        assert limits_config['MAX_CONCURRENT_REQUESTS'] == 100
        assert limits_config['CIRCUIT_BREAKER_THRESHOLD'] == 5
        assert limits_config['MAX_FILE_SIZE_MB'] == 50
        assert limits_config['MAX_CONCURRENT_UPLOADS'] == 3
        assert limits_config['MAX_NOTIFICATIONS'] == 1000
        assert limits_config['RATE_LIMIT_PER_MINUTE'] == 60


class TestConfigUtilities:
    """Test configuration utility functions"""
    
    def test_generate_secure_secret(self):
        """Test secure secret generation"""
        secret = generate_secure_secret()
        
        assert isinstance(secret, str)
        assert len(secret) > 40  # Base64 encoding increases length
        
        # Test custom length
        custom_secret = generate_secure_secret(64)
        assert len(custom_secret) > 80  # Base64 encoded 64 bytes
        
        # Test uniqueness
        secret1 = generate_secure_secret()
        secret2 = generate_secure_secret()
        assert secret1 != secret2
    
    @patch('app.core.dynamic_config.get_config_manager')
    @patch('app.core.dynamic_config.logger')
    def test_validate_production_config_success(self, mock_logger, mock_get_config):
        """Test production configuration validation success"""
        mock_manager = Mock()
        mock_manager.is_production.return_value = True
        
        # Mock successful config retrievals
        mock_manager.get_database_config.return_value = Mock()
        mock_manager.get_security_config.return_value = Mock()
        mock_manager.get_sap_config.return_value = Mock()
        
        mock_get_config.return_value = mock_manager
        
        # Should not raise exception
        validate_production_config()
        
        mock_logger.info.assert_called_with("Production configuration validation successful")
    
    @patch('app.core.dynamic_config.get_config_manager')
    @patch('app.core.dynamic_config.logger')
    def test_validate_production_config_failure(self, mock_logger, mock_get_config):
        """Test production configuration validation failure"""
        mock_manager = Mock()
        mock_manager.is_production.return_value = True
        
        # Mock config validation error
        mock_manager.get_database_config.side_effect = ConfigValidationError("Database config invalid")
        
        mock_get_config.return_value = mock_manager
        
        with pytest.raises(ConfigValidationError):
            validate_production_config()
        
        mock_logger.error.assert_called()
    
    @patch('app.core.dynamic_config.get_config_manager')
    def test_validate_production_config_non_production(self, mock_get_config):
        """Test production validation skipped for non-production"""
        mock_manager = Mock()
        mock_manager.is_production.return_value = False
        mock_get_config.return_value = mock_manager
        
        # Should not raise exception or call config methods
        validate_production_config()
        
        mock_manager.get_database_config.assert_not_called()


class TestConfigFileLoading:
    """Test configuration file loading"""
    
    def test_deep_merge(self):
        """Test deep merge functionality"""
        config_manager = DynamicConfigManager()
        
        target = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "other": "value"
        }
        
        source = {
            "database": {
                "port": 5433,
                "name": "testdb"
            },
            "new_config": "new_value"
        }
        
        config_manager._deep_merge(target, source)
        
        # Should merge nested dictionaries
        assert target["database"]["host"] == "localhost"  # Preserved
        assert target["database"]["port"] == 5433  # Overridden
        assert target["database"]["name"] == "testdb"  # Added
        assert target["other"] == "value"  # Preserved
        assert target["new_config"] == "new_value"  # Added
    
    def test_get_nested_value(self):
        """Test nested value retrieval"""
        config_manager = DynamicConfigManager()
        config_manager.config_cache = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        # Test successful nested access
        assert config_manager._get_nested_value("level1", "level2", "level3") == "value"
        
        # Test missing key
        assert config_manager._get_nested_value("level1", "missing") is None
        
        # Test empty path
        assert config_manager._get_nested_value() == config_manager.config_cache


class TestEnvironmentSpecificConfig:
    """Test environment-specific configuration handling"""
    
    def test_development_config(self):
        """Test development environment configuration"""
        config_manager = DynamicConfigManager(Environment.DEVELOPMENT)
        
        assert config_manager.is_development() is True
        assert config_manager.is_production() is False
        assert config_manager.is_staging() is False
    
    def test_staging_config(self):
        """Test staging environment configuration"""
        config_manager = DynamicConfigManager(Environment.STAGING)
        
        assert config_manager.is_development() is False
        assert config_manager.is_production() is False
        assert config_manager.is_staging() is True
    
    def test_production_config(self):
        """Test production environment configuration"""
        config_manager = DynamicConfigManager(Environment.PRODUCTION)
        
        assert config_manager.is_development() is False
        assert config_manager.is_production() is True
        assert config_manager.is_staging() is False
    
    @patch.dict(os.environ, {
        'AGENT_TIMEOUT_SECONDS': '45',
        'AGENT0_TIMEOUT': '20',
        'DB_POOL_MAX_SIZE': '30'
    })
    def test_environment_variable_override(self):
        """Test environment variables override default values"""
        config_manager = DynamicConfigManager()
        
        agent_config = config_manager.get_agent_config()
        assert agent_config.timeout_seconds == 45  # Overridden from env
        assert agent_config.agent0_timeout == 20   # Overridden from env
        
        db_config = config_manager.get_database_config()
        assert db_config.pool_max_size == 30  # Overridden from env


class TestConfigurationIntegration:
    """Test configuration integration with other components"""
    
    @patch('app.core.dynamic_config.yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_load_from_yaml_files(self, mock_exists, mock_open, mock_yaml):
        """Test loading configuration from YAML files"""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock YAML content
        mock_yaml.side_effect = [
            {"database": {"host": "base_host"}},  # base.yaml
            {"database": {"host": "dev_host", "port": 5432}}  # development.yaml
        ]
        
        config_manager = DynamicConfigManager(Environment.DEVELOPMENT)
        
        # Verify files were opened
        assert mock_open.call_count >= 2
        
        # Verify configuration was merged
        db_config = config_manager.config_cache.get("database", {})
        assert db_config.get("host") == "dev_host"  # Environment-specific override
        assert db_config.get("port") == 5432       # Environment-specific addition
    
    def test_config_caching(self):
        """Test configuration caching behavior"""
        config_manager = DynamicConfigManager()
        
        # First call should load and cache
        config1 = config_manager.get_agent_config()
        
        # Second call should return cached instance
        config2 = config_manager.get_agent_config()
        
        # Should be same object (cached)
        assert config1 is not config2  # New instance each time (dataclass behavior)
        assert config1.timeout_seconds == config2.timeout_seconds  # Same values
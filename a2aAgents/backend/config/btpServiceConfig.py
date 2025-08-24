"""
BTP Service Configuration for A2A Agents
Handles BTP service bindings and provides configuration for all agents
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BTPServiceConfig:
    """
    Configuration manager for BTP service bindings
    """
    
    def __init__(self):
        self.vcap_services = {}
        self.vcap_application = {}
        self._load_vcap_services()
    
    def _load_vcap_services(self):
        """Load VCAP_SERVICES from environment"""
        try:
            # Load VCAP_SERVICES
            vcap_services_str = os.getenv('VCAP_SERVICES', '{}')
            self.vcap_services = json.loads(vcap_services_str)
            
            # Load VCAP_APPLICATION
            vcap_application_str = os.getenv('VCAP_APPLICATION', '{}')
            self.vcap_application = json.loads(vcap_application_str)
            
            if self.vcap_services:
                logger.info("✅ BTP service bindings loaded from VCAP_SERVICES")
            else:
                logger.warning("⚠️ No VCAP_SERVICES found - using local development mode")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse VCAP_SERVICES: {e}")
            self.vcap_services = {}
            self.vcap_application = {}
    
    def _get_service_credentials(self, service_label: str, service_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get credentials for a specific service"""
        services = self.vcap_services.get(service_label, [])
        
        if not services:
            return None
        
        # If service_name is specified, find by name
        if service_name:
            for service in services:
                if service.get('name') == service_name:
                    return service.get('credentials', {})
        
        # Return first service if no name specified
        return services[0].get('credentials', {}) if services else None
    
    def get_hana_config(self) -> Dict[str, Any]:
        """Get HANA database configuration"""
        # Try HANA Cloud first, then legacy hana service
        hana_creds = (self._get_service_credentials('hana', 'a2a-network-db') or 
                     self._get_service_credentials('hanatrial', 'a2a-network-db') or
                     self._get_service_credentials('hana') or
                     self._get_service_credentials('hanatrial'))
        
        if hana_creds:
            # BTP HANA Cloud configuration
            config = {
                'address': hana_creds.get('host', hana_creds.get('hostname')),
                'port': int(hana_creds.get('port', hana_creds.get('sql_port', 443))),
                'user': hana_creds.get('user', hana_creds.get('hdi_user')),
                'password': hana_creds.get('password', hana_creds.get('hdi_password')),
                'database': hana_creds.get('database', hana_creds.get('db_name')),
                'schema': hana_creds.get('schema', os.getenv('HANA_SCHEMA', 'A2A_AGENTS')),
                'encrypt': True,
                'ssl_validate_certificate': True,
                'auto_commit': False,
                
                # Connection pool settings optimized for BTP
                'pool_size': int(os.getenv('HANA_POOL_SIZE', '15')),
                'max_overflow': int(os.getenv('HANA_MAX_OVERFLOW', '25')),
                'pool_timeout': int(os.getenv('HANA_POOL_TIMEOUT', '30')),
                'pool_recycle': int(os.getenv('HANA_POOL_RECYCLE', '3600')),
                'pool_pre_ping': True,
                
                # HANA-specific optimizations for BTP
                'isolation_level': 'READ_COMMITTED',
                'connection_timeout': 30,
                'compress': True,
                'fetch_size': 1000,
                
                # Enterprise features
                'backup_enabled': True,
                'monitoring_enabled': True,
                'query_timeout': 300
            }
            
            logger.info(f"✅ HANA configuration loaded from BTP service binding: {config['address']}")
            return config
        
        else:
            # Local development fallback
            logger.warning("⚠️ Using HANA local development configuration")
            return {
                'address': os.getenv('HANA_HOST', 'localhost'),
                'port': int(os.getenv('HANA_PORT', '30015')),
                'user': os.getenv('HANA_USER', 'SYSTEM'),
                'password': os.getenv('HANA_PASSWORD', ''),
                'database': os.getenv('HANA_DATABASE', 'A2A'),
                'schema': os.getenv('HANA_SCHEMA', 'A2A_AGENTS'),
                'encrypt': False,
                'ssl_validate_certificate': False,
                'auto_commit': False,
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'pool_pre_ping': True
            }
    
    def get_xsuaa_config(self) -> Dict[str, Any]:
        """Get XSUAA authentication configuration"""
        xsuaa_creds = self._get_service_credentials('xsuaa', 'a2a-network-xsuaa')
        
        if xsuaa_creds:
            config = {
                'client_id': xsuaa_creds.get('clientid'),
                'client_secret': xsuaa_creds.get('clientsecret'),
                'url': xsuaa_creds.get('url'),
                'uaa_domain': xsuaa_creds.get('uaadomain'),
                'xs_app_name': xsuaa_creds.get('xsappname'),
                'identity_zone': xsuaa_creds.get('identityzone'),
                'identity_zone_id': xsuaa_creds.get('identityzoneid'),
                'tenant_id': xsuaa_creds.get('tenantid'),
                'tenant_mode': xsuaa_creds.get('tenantmode', 'dedicated'),
                'verification_key': xsuaa_creds.get('verificationkey'),
                'trusted_client_id_suffix': xsuaa_creds.get('trustedclientidsuffix'),
                
                # JWT settings
                'jwt_algorithm': 'RS256',
                'jwt_access_token_expires': 3600,
                'jwt_refresh_token_expires': 86400 * 7
            }
            
            logger.info("✅ XSUAA configuration loaded from BTP service binding")
            return config
        
        else:
            # Local development fallback
            logger.warning("⚠️ Using XSUAA local development configuration")
            return {
                'client_id': os.getenv('XSUAA_CLIENT_ID', 'local-client'),
                'client_secret': os.getenv('XSUAA_CLIENT_SECRET', 'local-secret'),
                'url': os.getenv('XSUAA_URL', 'http://localhost:8080/uaa'),
                'xs_app_name': os.getenv('XSUAA_XSAPPNAME', 'a2a-agents-local'),
                'tenant_mode': 'dedicated',
                'jwt_algorithm': 'HS256',
                'jwt_access_token_expires': 3600
            }
    
    def get_destination_config(self) -> Optional[Dict[str, Any]]:
        """Get Destination service configuration"""
        dest_creds = self._get_service_credentials('destination', 'a2a-network-destination-service')
        
        if dest_creds:
            config = {
                'uri': dest_creds.get('uri'),
                'client_id': dest_creds.get('clientid'),
                'client_secret': dest_creds.get('clientsecret'),
                'url': dest_creds.get('url'),
                'token_service_url': dest_creds.get('tokenServiceUrl'),
                'token_service_url_pattern': dest_creds.get('tokenServiceURLPattern')
            }
            
            logger.info("✅ Destination service configuration loaded")
            return config
        
        logger.warning("⚠️ Destination service not bound - external system access may be limited")
        return None
    
    def get_redis_config(self) -> Optional[Dict[str, Any]]:
        """Get Redis cache configuration"""
        redis_creds = self._get_service_credentials('redis-cache', 'a2a-network-redis')
        
        if redis_creds:
            config = {
                'host': redis_creds.get('hostname', redis_creds.get('host')),
                'port': int(redis_creds.get('port', 6379)),
                'password': redis_creds.get('password'),
                'tls': redis_creds.get('tls_enabled', False),
                'db': 0,
                
                # Redis optimization for BTP
                'socket_connect_timeout': 5,
                'socket_timeout': 5,
                'retry_on_timeout': True,
                'max_connections': 50,
                'connection_pool_kwargs': {
                    'retry_on_timeout': True,
                    'socket_keepalive': True,
                    'socket_keepalive_options': {},
                    'health_check_interval': 30
                }
            }
            
            logger.info("✅ Redis cache configuration loaded from BTP service binding")
            return config
        
        logger.warning("⚠️ Redis service not bound - caching will be disabled")
        return None
    
    def get_service_manager_config(self) -> Optional[Dict[str, Any]]:
        """Get Service Manager configuration"""
        sm_creds = self._get_service_credentials('service-manager', 'a2a-network-service-manager')
        
        if sm_creds:
            config = {
                'url': sm_creds.get('url'),
                'client_id': sm_creds.get('clientid'),
                'client_secret': sm_creds.get('clientsecret'),
                'token_url': sm_creds.get('token_url', sm_creds.get('url') + '/oauth/token')
            }
            
            logger.info("✅ Service Manager configuration loaded")
            return config
        
        logger.warning("⚠️ Service Manager not bound - service-to-service calls may fail")
        return None
    
    def get_logging_config(self) -> Optional[Dict[str, Any]]:
        """Get Application Logging configuration"""
        log_creds = self._get_service_credentials('application-logs', 'a2a-network-application-logging')
        
        if log_creds:
            config = {
                'endpoint': log_creds.get('endpoint'),
                'user': log_creds.get('user'),
                'password': log_creds.get('password'),
                'level': os.getenv('LOG_LEVEL', 'info'),
                'format': os.getenv('LOG_FORMAT', 'json'),
                'retention_days': int(os.getenv('LOG_RETENTION_DAYS', '7'))
            }
            
            logger.info("✅ Application logging configuration loaded")
            return config
        
        logger.warning("⚠️ Application logging service not bound - using console logging")
        return None
    
    def get_alert_notification_config(self) -> Optional[Dict[str, Any]]:
        """Get Alert Notification configuration"""
        alert_creds = self._get_service_credentials('alert-notification', 'a2a-network-alert-notification')
        
        if alert_creds:
            config = {
                'url': alert_creds.get('url'),
                'client_id': alert_creds.get('client_id'),
                'client_secret': alert_creds.get('client_secret'),
                'oauth2_url': alert_creds.get('oauth2_url')
            }
            
            logger.info("✅ Alert notification configuration loaded")
            return config
        
        logger.warning("⚠️ Alert notification service not bound - alerts will be logged only")
        return None
    
    def get_connectivity_config(self) -> Optional[Dict[str, Any]]:
        """Get Connectivity service configuration for on-premise systems"""
        conn_creds = self._get_service_credentials('connectivity', 'a2a-network-connectivity-service')
        
        if conn_creds:
            config = {
                'onpremise_proxy_host': conn_creds.get('onpremise_proxy_host'),
                'onpremise_proxy_port': conn_creds.get('onpremise_proxy_port'),
                'onpremise_proxy_ldap_port': conn_creds.get('onpremise_proxy_ldap_port'),
                'onpremise_proxy_rfc_port': conn_creds.get('onpremise_proxy_rfc_port'),
                'onpremise_socks5_proxy_port': conn_creds.get('onpremise_socks5_proxy_port')
            }
            
            logger.info("✅ Connectivity service configuration loaded")
            return config
        
        logger.warning("⚠️ Connectivity service not bound - on-premise access not available")
        return None
    
    def get_application_info(self) -> Dict[str, Any]:
        """Get application information from VCAP_APPLICATION"""
        if self.vcap_application:
            return {
                'name': self.vcap_application.get('name', 'a2a-agents'),
                'space_name': self.vcap_application.get('space_name'),
                'space_id': self.vcap_application.get('space_id'),
                'organization_name': self.vcap_application.get('organization_name'),
                'organization_id': self.vcap_application.get('organization_id'),
                'instance_id': self.vcap_application.get('instance_id'),
                'instance_index': self.vcap_application.get('instance_index', 0),
                'host': self.vcap_application.get('host', '0.0.0.0'),
                'port': self.vcap_application.get('port', 8080),
                'uris': self.vcap_application.get('uris', []),
                'version': self.vcap_application.get('version', '1.0.0')
            }
        
        return {
            'name': 'a2a-agents-local',
            'instance_index': 0,
            'host': '0.0.0.0',
            'port': 8080,
            'version': '1.0.0'
        }
    
    def is_cloud_foundry(self) -> bool:
        """Check if running in Cloud Foundry environment"""
        return bool(self.vcap_services or self.vcap_application)
    
    def validate_critical_services(self) -> bool:
        """Validate that critical services are bound"""
        critical_services = {
            'HANA Database': self.get_hana_config(),
            'XSUAA Authentication': self.get_xsuaa_config()
        }
        
        missing_services = []
        for service_name, config in critical_services.items():
            if not config or (isinstance(config, dict) and not config.get('address') and not config.get('client_id')):
                missing_services.append(service_name)
        
        if missing_services:
            logger.error(f"❌ Critical services not properly configured: {', '.join(missing_services)}")
            return False
        
        logger.info("✅ All critical BTP services validated")
        return True
    
    def get_service_binding_status(self) -> Dict[str, str]:
        """Get status of all service bindings"""
        status = {}
        
        service_checks = {
            'HANA Database': self.get_hana_config,
            'XSUAA Authentication': self.get_xsuaa_config,
            'Destination Service': self.get_destination_config,
            'Redis Cache': self.get_redis_config,
            'Service Manager': self.get_service_manager_config,
            'Application Logging': self.get_logging_config,
            'Alert Notification': self.get_alert_notification_config,
            'Connectivity Service': self.get_connectivity_config
        }
        
        for service_name, check_func in service_checks.items():
            try:
                config = check_func()
                if config:
                    status[service_name] = "✅ BOUND"
                else:
                    status[service_name] = "⚠️ NOT_BOUND"
            except Exception as e:
                status[service_name] = f"❌ ERROR: {str(e)}"
        
        return status


# Global configuration instance
btp_config = BTPServiceConfig()
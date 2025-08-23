"""
Minimal BTP Configuration - Local First, BTP Optional
Simple approach that works locally and adds BTP services when available
"""

import os
import json
import logging
from typing import Dict, Any, Optional


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class MinimalBTPConfig:
    """
    Minimal configuration that works locally and uses BTP when available
    No over-engineering - just the essentials
    """
    
    def __init__(self):
        self.is_btp = self._detect_btp_environment()
        self.services = self._load_services()
    
    def _detect_btp_environment(self) -> bool:
        """Simple BTP detection"""
        return bool(os.getenv('VCAP_SERVICES') or os.getenv('VCAP_APPLICATION'))
    
    def _load_services(self) -> Dict[str, Any]:
        """Load services - BTP if available, local fallback otherwise"""
        if self.is_btp:
            return self._load_btp_services()
        else:
            return self._load_local_services()
    
    def _load_btp_services(self) -> Dict[str, Any]:
        """Load from BTP VCAP_SERVICES"""
        try:
            vcap_services = json.loads(os.getenv('VCAP_SERVICES', '{}'))
            logger.info("✅ Running on BTP - using service bindings")
            
            services = {}
            
            # HANA - get first available
            hana_services = (vcap_services.get('hana') or 
                           vcap_services.get('hanatrial') or [])
            if hana_services:
                creds = hana_services[0]['credentials']
                services['hana'] = {
                    'host': creds.get('host'),
                    'port': creds.get('port', 443),
                    'user': creds.get('user'),
                    'password': creds.get('password'),
                    'encrypt': True,
                    'schema': creds.get('schema', 'A2A_AGENTS')
                }
            
            # XSUAA - get first available
            xsuaa_services = vcap_services.get('xsuaa', [])
            if xsuaa_services:
                creds = xsuaa_services[0]['credentials']
                services['xsuaa'] = {
                    'url': creds.get('url'),
                    'client_id': creds.get('clientid'),
                    'client_secret': creds.get('clientsecret')
                }
            
            # Redis - optional
            redis_services = vcap_services.get('redis-cache', [])
            if redis_services:
                creds = redis_services[0]['credentials']
                services['redis'] = {
                    'host': creds.get('hostname'),
                    'port': creds.get('port', 6379),
                    'password': creds.get('password')
                }
            
            return services
            
        except Exception as e:
            logger.warning(f"Failed to load BTP services: {e}")
            return self._load_local_services()
    
    def _load_local_services(self) -> Dict[str, Any]:
        """Local development configuration"""
        logger.info("🔧 Running locally - using environment variables")
        
        return {
            'hana': {
                'host': os.getenv('HANA_HOST', 'localhost'),
                'port': int(os.getenv('HANA_PORT', '30015')),
                'user': os.getenv('HANA_USER', 'SYSTEM'),
                'password': os.getenv('HANA_PASSWORD', ''),
                'encrypt': False,  # Local development
                'schema': os.getenv('HANA_SCHEMA', 'A2A_AGENTS')
            },
            'xsuaa': {
                'url': os.getenv('XSUAA_URL', os.getenv("A2A_GATEWAY_URL")),
                'client_id': os.getenv('XSUAA_CLIENT_ID', 'local-client'),
                'client_secret': os.getenv('XSUAA_CLIENT_SECRET', 'local-secret')
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', '6379')),
                'password': os.getenv('REDIS_PASSWORD', '')
            } if os.getenv('REDIS_HOST') else None
        }
    
    def get_hana_config(self) -> Dict[str, Any]:
        """Get HANA configuration"""
        hana = self.services.get('hana')
        if not hana:
            raise ValueError("HANA configuration not available")
        return hana
    
    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration"""
        if self.is_btp:
            return self.services.get('xsuaa', {})
        else:
            # Local development - simple auth
            return {
                'local_mode': True,
                'bypass_auth': os.getenv('BYPASS_AUTH', 'true').lower() == 'true'
            }
    
    def get_cache_config(self) -> Optional[Dict[str, Any]]:
        """Get cache configuration if available"""
        return self.services.get('redis')
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if a service is available"""
        return service_name in self.services and self.services[service_name] is not None


# Global instance
config = MinimalBTPConfig()
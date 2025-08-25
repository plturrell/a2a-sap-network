"""
Hybrid A2A Network Client
Maintains A2A protocol compliance while allowing secure HTTP for approved external services
"""

import os
import asyncio
import json
import logging
import fnmatch
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CommunicationType(str, Enum):
    """Types of communication in hybrid approach"""
    A2A_INTERNAL = "a2a_internal"  # Agent-to-agent via blockchain
    EXTERNAL_HTTP = "external_http"  # External services via HTTP
    HYBRID_WRAPPED = "hybrid_wrapped"  # HTTP wrapped in A2A messages


class ExternalDomainConfig(BaseModel):
    """Configuration for allowed external domains"""
    domain_pattern: str
    description: str
    allowed_methods: List[str] = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    requires_auth: bool = True
    max_timeout: int = 30


class HybridNetworkClient:
    """
    Hybrid A2A Network Client
    
    This client maintains A2A protocol compliance by:
    1. Using A2A blockchain messaging for agent-to-agent communication
    2. Allowing secure HTTP for approved external services (like SAP BTP)
    3. Wrapping external HTTP calls in A2A message format for consistency
    4. Logging all communication for audit compliance
    """

    def __init__(self, agent_id: str, blockchain_client: Optional[Any] = None):
        self.agent_id = agent_id
        self.blockchain_client = blockchain_client
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize allowed external domains
        self.external_domains = self._initialize_external_domains()
        
        # HTTP client for external services
        self.http_client_config = {
            "timeout": 30.0,
            "verify": True,
            "follow_redirects": True,
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100)
        }
        
        logger.info(f"Hybrid Network Client initialized for agent: {agent_id}")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load hybrid network configuration"""
        return {
            "enable_external_http": os.getenv("A2A_ENABLE_EXTERNAL_HTTP", "true").lower() == "true",
            "strict_domain_checking": os.getenv("A2A_STRICT_DOMAIN_CHECK", "true").lower() == "true",
            "log_all_requests": os.getenv("A2A_LOG_ALL_REQUESTS", "true").lower() == "true",
            "require_a2a_wrapper": os.getenv("A2A_REQUIRE_WRAPPER", "true").lower() == "true"
        }

    def _initialize_external_domains(self) -> List[ExternalDomainConfig]:
        """Initialize allowed external domains"""
        return [
            # SAP BTP Authentication Services
            ExternalDomainConfig(
                domain_pattern="*.authentication.*.hana.ondemand.com",
                description="SAP BTP XSUAA Authentication",
                allowed_methods=["POST"],
                requires_auth=True,
                max_timeout=30
            ),
            ExternalDomainConfig(
                domain_pattern="*.authentication.sap.hana.ondemand.com",
                description="SAP BTP XSUAA Authentication - General",
                allowed_methods=["POST"],
                requires_auth=True,
                max_timeout=30
            ),
            
            # SAP BTP Destination Service
            ExternalDomainConfig(
                domain_pattern="*.dest-configuration.*.hana.ondemand.com",
                description="SAP BTP Destination Service",
                allowed_methods=["GET", "POST"],
                requires_auth=True,
                max_timeout=45
            ),
            
            # SAP BTP Alert Notification Service
            ExternalDomainConfig(
                domain_pattern="*.notification.*.hana.ondemand.com",
                description="SAP BTP Alert Notification",
                allowed_methods=["GET", "POST"],
                requires_auth=True,
                max_timeout=30
            ),
            
            # SAP BTP Connectivity Service
            ExternalDomainConfig(
                domain_pattern="*.connectivity.*.hana.ondemand.com",
                description="SAP BTP Connectivity Service",
                allowed_methods=["GET", "POST"],
                requires_auth=True,
                max_timeout=60
            ),
            
            # Custom domains from environment
            *self._load_custom_domains()
        ]

    def _load_custom_domains(self) -> List[ExternalDomainConfig]:
        """Load custom allowed domains from environment"""
        custom_domains = []
        
        domains_config = os.getenv("A2A_ALLOWED_EXTERNAL_DOMAINS")
        if domains_config:
            try:
                domains_data = json.loads(domains_config)
                for domain_data in domains_data:
                    custom_domains.append(ExternalDomainConfig(**domain_data))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load custom domains config: {e}")
        
        return custom_domains

    def _determine_communication_type(self, target: str, endpoint: Optional[str] = None) -> CommunicationType:
        """Determine the type of communication needed"""
        
        # If target looks like an agent ID (no dots, no http), it's internal A2A
        if not any(char in target for char in ['.', '/', 'http']):
            return CommunicationType.A2A_INTERNAL
        
        # If endpoint is provided, check if it's an allowed external domain
        if endpoint and self._is_external_domain_allowed(endpoint):
            if self.config["require_a2a_wrapper"]:
                return CommunicationType.HYBRID_WRAPPED
            else:
                return CommunicationType.EXTERNAL_HTTP
        
        # Default to A2A internal
        return CommunicationType.A2A_INTERNAL

    def _is_external_domain_allowed(self, url: str) -> bool:
        """Check if external domain is allowed"""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            for domain_config in self.external_domains:
                if fnmatch.fnmatch(domain, domain_config.domain_pattern):
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking domain allowance: {e}")
            return False

    async def send_a2a_message(
        self, 
        to_agent: str, 
        message: Dict[str, Any], 
        message_type: str = "GENERAL"
    ) -> Dict[str, Any]:
        """Send message using appropriate communication method"""
        
        comm_type = self._determine_communication_type(to_agent)
        
        if comm_type == CommunicationType.A2A_INTERNAL:
            return await self._send_internal_a2a_message(to_agent, message, message_type)
        elif comm_type == CommunicationType.HYBRID_WRAPPED:
            return await self._send_hybrid_wrapped_message(to_agent, message, message_type)
        else:
            raise ValueError(f"Unsupported communication type: {comm_type}")

    async def _send_internal_a2a_message(
        self, 
        to_agent: str, 
        message: Dict[str, Any], 
        message_type: str
    ) -> Dict[str, Any]:
        """Send internal A2A message through blockchain"""
        
        if not self.blockchain_client:
            # Fallback for development/testing
            logger.warning(f"No blockchain client available, simulating A2A message to {to_agent}")
            return {
                "success": True,
                "message_id": f"sim_{int(datetime.utcnow().timestamp())}",
                "type": "simulated_a2a",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            # Use actual blockchain client
            result = await self.blockchain_client.send_message(
                to_address=to_agent,
                message=message,
                message_type=message_type
            )
            
            if self.config["log_all_requests"]:
                logger.info(f"A2A message sent: {self.agent_id} -> {to_agent} ({message_type})")
            
            return {
                "success": True,
                "tx_hash": result.get("tx_hash"),
                "message_id": result.get("message_id"),
                "type": "blockchain_a2a",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "blockchain_a2a",
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _send_hybrid_wrapped_message(
        self, 
        endpoint: str, 
        message: Dict[str, Any], 
        message_type: str
    ) -> Dict[str, Any]:
        """Send HTTP request wrapped in A2A message format"""
        
        if not self.config["enable_external_http"]:
            raise RuntimeError("External HTTP is disabled in A2A configuration")
        
        # Validate domain
        if self.config["strict_domain_checking"] and not self._is_external_domain_allowed(endpoint):
            raise RuntimeError(f"Domain not allowed for external HTTP: {endpoint}")
        
        try:
            # Extract HTTP details from message
            http_method = message.get("method", "GET")
            headers = message.get("headers", {})
            data = message.get("data")
            params = message.get("params")
            timeout = message.get("timeout", 30)
            
            # Make HTTP request
            async with httpx.AsyncClient(**self.http_client_config) as client:
                response = await client.request(
                    method=http_method,
                    url=endpoint,
                    headers=headers,
                    json=data if isinstance(data, dict) else None,
                    data=data if not isinstance(data, dict) else None,
                    params=params,
                    timeout=timeout
                )
                
                # Parse response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                # Wrap in A2A message format
                wrapped_response = {
                    "success": True,
                    "http_status": response.status_code,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "type": "hybrid_wrapped_http",
                    "endpoint": endpoint,
                    "method": http_method,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if self.config["log_all_requests"]:
                    logger.info(f"Hybrid HTTP request: {self.agent_id} -> {endpoint} ({http_method}) -> {response.status_code}")
                
                return wrapped_response
                
        except Exception as e:
            logger.error(f"Hybrid HTTP request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "type": "hybrid_wrapped_http",
                "endpoint": endpoint,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def send_sap_btp_request(
        self,
        service_type: str,
        operation: str,
        endpoint: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Specialized method for SAP BTP service requests
        Uses hybrid approach with HTTP for external SAP services
        """
        
        # Validate SAP BTP domain
        if not self._is_external_domain_allowed(endpoint):
            raise RuntimeError(f"SAP BTP endpoint not in allowed domains: {endpoint}")
        
        # Create message in A2A format
        sap_message = {
            "service_type": service_type,
            "operation": operation,
            "method": method.upper(),
            "headers": headers or {},
            "data": data,
            "params": params,
            "timeout": timeout,
            "from_agent": self.agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send as hybrid wrapped message
        result = await self._send_hybrid_wrapped_message(endpoint, sap_message, f"SAP_BTP_{service_type}")
        
        # Add SAP BTP specific metadata
        if result.get("success"):
            result["sap_btp_service"] = service_type
            result["sap_btp_operation"] = operation
        
        return result

    def add_allowed_domain(self, domain_config: ExternalDomainConfig):
        """Add new allowed external domain"""
        self.external_domains.append(domain_config)
        logger.info(f"Added allowed external domain: {domain_config.domain_pattern}")

    def is_domain_allowed(self, url: str) -> bool:
        """Public method to check if domain is allowed"""
        return self._is_external_domain_allowed(url)

    def get_allowed_domains(self) -> List[str]:
        """Get list of allowed domain patterns"""
        return [config.domain_pattern for config in self.external_domains]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for hybrid network client"""
        return {
            "agent_id": self.agent_id,
            "communication_types_supported": [
                CommunicationType.A2A_INTERNAL.value,
                CommunicationType.HYBRID_WRAPPED.value
            ],
            "blockchain_client_available": self.blockchain_client is not None,
            "external_http_enabled": self.config["enable_external_http"],
            "allowed_domains_count": len(self.external_domains),
            "configuration": self.config,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global client factory
_hybrid_clients: Dict[str, HybridNetworkClient] = {}


def get_hybrid_network_client(agent_id: str, blockchain_client: Optional[Any] = None) -> HybridNetworkClient:
    """Get or create hybrid network client for agent"""
    if agent_id not in _hybrid_clients:
        _hybrid_clients[agent_id] = HybridNetworkClient(agent_id, blockchain_client)
    
    return _hybrid_clients[agent_id]


def create_sap_btp_client(agent_id: str) -> HybridNetworkClient:
    """Create hybrid client specifically configured for SAP BTP services"""
    client = get_hybrid_network_client(agent_id)
    
    # Add any additional SAP BTP specific configuration
    logger.info(f"Created SAP BTP hybrid client for agent: {agent_id}")
    
    return client
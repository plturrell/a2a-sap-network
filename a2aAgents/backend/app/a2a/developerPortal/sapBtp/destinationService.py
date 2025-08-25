"""
SAP Destination Service Integration
Provides secure connectivity to external services through SAP BTP Destination Service
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

from pydantic import BaseModel, Field
from fastapi import HTTPException

# A2A Protocol imports - Hybrid approach
from ...core.hybridNetworkClient import create_sap_btp_client

logger = logging.getLogger(__name__)


class AuthenticationType(str, Enum):
    """Destination authentication types"""
    NO_AUTHENTICATION = "NoAuthentication"
    BASIC_AUTHENTICATION = "BasicAuthentication"
    OAUTH2_CLIENT_CREDENTIALS = "OAuth2ClientCredentials"
    OAUTH2_USER_TOKEN_EXCHANGE = "OAuth2UserTokenExchange"
    OAUTH2_SAML_BEARER_ASSERTION = "OAuth2SAMLBearerAssertion"
    SAML_ASSERTION = "SAMLAssertion"
    PRINCIPAL_PROPAGATION = "PrincipalPropagation"


class ProxyType(str, Enum):
    """Destination proxy types"""
    INTERNET = "Internet"
    ON_PREMISE = "OnPremise"


class DestinationConfig(BaseModel):
    """Destination configuration"""
    name: str
    url: str
    description: str = ""
    authentication: AuthenticationType = AuthenticationType.NO_AUTHENTICATION
    proxy_type: ProxyType = ProxyType.INTERNET

    # Authentication details
    user: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_service_url: Optional[str] = None

    # Additional properties
    additional_properties: Dict[str, str] = Field(default_factory=dict)

    # Connection settings
    timeout: int = 30000  # milliseconds
    max_connections: int = 100


class DestinationInfo(BaseModel):
    """Destination information retrieved from service"""
    name: str
    url: str
    authentication: AuthenticationType
    proxy_type: ProxyType
    properties: Dict[str, Any] = Field(default_factory=dict)
    auth_tokens: Dict[str, str] = Field(default_factory=dict)


class DestinationService:
    """SAP Destination Service client"""

    def __init__(self, service_config: Dict[str, Any]):
        self.service_config = service_config
        self.service_url = service_config.get("uri", "")
        self.client_id = service_config.get("clientid", "")
        self.client_secret = service_config.get("clientsecret", "")
        self.token_url = service_config.get("url", "") + "/oauth/token"

        # Cache for destinations and tokens
        self.destination_cache: Dict[str, DestinationInfo] = {}
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=10)

        # Initialize hybrid network client for A2A compliance with external HTTP
        self.hybrid_client = create_sap_btp_client(f"destination_service_{id(self)}")
        
        logger.info("SAP Destination Service initialized with A2A hybrid compliance")

    async def get_access_token(self) -> str:
        """Get OAuth2 access token for Destination Service through A2A protocol"""
        try:
            cache_key = "destination_service_token"

            # Check cache
            if cache_key in self.token_cache:
                token_info = self.token_cache[cache_key]
                if datetime.utcnow() < token_info["expires_at"]:
                    return token_info["access_token"]

            # Request new token through A2A protocol
            auth_header = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            data = {
                "grant_type": "client_credentials"
            }

            # A2A Protocol: Use hybrid approach for SAP BTP services
            response = await self.hybrid_client.send_sap_btp_request(
                service_type="oauth_token",
                operation="get_access_token", 
                endpoint=self.token_url,
                method="POST",
                headers=headers,
                data=data
            )

            if response.get("success") and response.get("data"):
                token_data = response["data"]
                access_token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)

                # Cache token
                self.token_cache[cache_key] = {
                    "access_token": access_token,
                    "expires_at": datetime.utcnow() + timedelta(seconds=expires_in - 60)
                }

                return access_token
            else:
                raise Exception(f"Hybrid A2A token request failed: {response.get('error')}")

        except Exception as e:
            logger.error(f"Failed to get Destination Service access token: {e}")
            raise HTTPException(status_code=500, detail="Failed to authenticate with Destination Service")

    async def get_destination(self, destination_name: str, user_token: Optional[str] = None) -> DestinationInfo:
        """Get destination configuration from Destination Service"""
        try:
            # Check cache
            cache_key = f"{destination_name}_{user_token or 'system'}"
            if cache_key in self.destination_cache:
                return self.destination_cache[cache_key]

            # Get service access token
            access_token = await self.get_access_token()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            # Add user token for user token exchange
            if user_token:
                headers["X-User-Token"] = user_token

            url = f"{self.service_url}/destination-configuration/v1/destinations/{destination_name}"
            
            # A2A Protocol: Use hybrid approach for SAP BTP services
            response = await self.hybrid_client.send_sap_btp_request(
                service_type="destination_config",
                operation="get_destination",
                endpoint=url,
                method="GET",
                headers=headers
            )

            if not response.get("success"):
                raise Exception(f"Hybrid A2A destination request failed: {response.get('error')}")

            dest_data = response.get("data")

                # Parse destination info
                destination_info = DestinationInfo(
                    name=dest_data["Name"],
                    url=dest_data["URL"],
                    authentication=AuthenticationType(dest_data.get("Authentication", "NoAuthentication")),
                    proxy_type=ProxyType(dest_data.get("ProxyType", "Internet")),
                    properties=dest_data
                )

                # Get authentication tokens if needed
                if destination_info.authentication != AuthenticationType.NO_AUTHENTICATION:
                    auth_tokens = await self._get_destination_auth_tokens(
                        destination_name,
                        destination_info.authentication,
                        user_token
                    )
                    destination_info.auth_tokens = auth_tokens

                # Cache destination
                self.destination_cache[cache_key] = destination_info

                return destination_info

        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Destination '{destination_name}' not found")
            else:
                logger.error(f"Failed to get destination {destination_name}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve destination")
        except Exception as e:
            logger.error(f"Error getting destination {destination_name}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve destination")

    async def _get_destination_auth_tokens(
        self,
        destination_name: str,
        auth_type: AuthenticationType,
        user_token: Optional[str] = None
    ) -> Dict[str, str]:
        """Get authentication tokens for destination"""
        try:
            access_token = await self.get_access_token()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            if user_token:
                headers["X-User-Token"] = user_token

            url = f"{self.service_url}/destination-configuration/v1/destinations/{destination_name}/authTokens"
            
            # A2A Protocol: Use hybrid approach for SAP BTP services
            response = await self.hybrid_client.send_sap_btp_request(
                service_type="destination_auth",
                operation="get_auth_tokens",
                endpoint=url,
                method="POST",
                headers=headers
            )

            if response.get("success"):
                return response.get("data", {})
            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to get auth tokens for destination {destination_name}: {e}")
            return {}

    async def list_destinations(self) -> List[str]:
        """List all available destinations"""
        try:
            access_token = await self.get_access_token()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            url = f"{self.service_url}/destination-configuration/v1/destinations"
            
            # A2A Protocol: Use hybrid approach for SAP BTP services
            response = await self.hybrid_client.send_sap_btp_request(
                service_type="destinations_list",
                operation="list_destinations",
                endpoint=url,
                method="GET",
                headers=headers
            )

            if response.get("success") and response.get("data"):
                destinations_data = response.get("data")
                return [dest.get("Name", "") for dest in destinations_data] if isinstance(destinations_data, list) else []
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to list destinations: {e}")
            return []


# Global destination service instance
destination_service: Optional[DestinationService] = None


def get_destination_service() -> DestinationService:
    """Get destination service instance"""
    if destination_service is None:
        raise HTTPException(status_code=500, detail="Destination service not initialized")
    return destination_service


def initialize_destination_service(service_config: Dict[str, Any]):
    """Initialize the global destination service"""
    global destination_service
    destination_service = DestinationService(service_config)
    logger.info("Destination Service initialized globally")

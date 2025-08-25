"""
A2A Compliant SAP BTP Service Agent
Provides SAP BTP integration through A2A blockchain messaging protocol
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

# A2A Protocol imports
from ...core.networkClient import A2ANetworkClient
from ...sdk.blockchainIntegration import BlockchainIntegrationMixin
from ...sdk.agentBase import A2AAgent
from ...sdk.types import A2AMessage, MessageType, AgentCapability

logger = logging.getLogger(__name__)


class SAPBTPServiceType(str, Enum):
    """SAP BTP service types"""
    DESTINATION_SERVICE = "destination_service"
    XSUAA_SERVICE = "xsuaa_service"  
    NOTIFICATION_SERVICE = "notification_service"
    HANA_SERVICE = "hana_service"


class A2ASAPBTPRequest(BaseModel):
    """A2A compliant SAP BTP request"""
    service_type: SAPBTPServiceType
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    authentication: Optional[Dict[str, str]] = None


class A2ASAPBTPResponse(BaseModel):
    """A2A compliant SAP BTP response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    service_type: SAPBTPServiceType
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class A2ASAPBTPAgent(A2AAgent, BlockchainIntegrationMixin):
    """
    A2A Protocol Compliant SAP BTP Service Agent
    
    This agent handles SAP BTP service interactions through A2A blockchain
    messaging, eliminating direct HTTP calls for protocol compliance.
    """

    def __init__(self, agent_id: str = "sap_btp_agent", **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # Agent capabilities
        self.capabilities = [
            AgentCapability.DATA_PROCESSING,
            AgentCapability.EXTERNAL_API_INTEGRATION,
            AgentCapability.AUTHENTICATION,
            AgentCapability.SERVICE_ORCHESTRATION
        ]
        
        # SAP BTP configuration cache
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=10)
        
        # A2A Network client
        self.network_client = A2ANetworkClient(agent_id=self.agent_id)
        
        logger.info(f"A2A SAP BTP Agent initialized: {agent_id}")

    async def initialize(self):
        """Initialize the A2A SAP BTP agent"""
        await super().initialize()
        
        # Register capabilities with blockchain
        await self.register_capability(AgentCapability.EXTERNAL_API_INTEGRATION)
        await self.register_capability(AgentCapability.AUTHENTICATION)
        
        logger.info("A2A SAP BTP Agent initialized successfully")

    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle incoming A2A messages"""
        try:
            if message.payload.get("request_type") == "sap_btp_service":
                request = A2ASAPBTPRequest(**message.payload["service_request"])
                response = await self._handle_sap_btp_request(request)
                
                # Return A2A response message
                return A2AMessage(
                    id=f"sap_btp_response_{int(datetime.utcnow().timestamp())}",
                    type=MessageType.RESPONSE,
                    sender=self.agent_id,
                    receiver=message.sender,
                    in_reply_to=message.id,
                    payload={
                        "response_type": "sap_btp_service",
                        "service_response": response.dict()
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Error handling SAP BTP message: {e}")
            return A2AMessage(
                id=f"sap_btp_error_{int(datetime.utcnow().timestamp())}",
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                in_reply_to=message.id,
                payload={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )

    async def _handle_sap_btp_request(self, request: A2ASAPBTPRequest) -> A2ASAPBTPResponse:
        """Handle SAP BTP service request through A2A protocol"""
        try:
            if request.service_type == SAPBTPServiceType.DESTINATION_SERVICE:
                return await self._handle_destination_service(request)
            elif request.service_type == SAPBTPServiceType.XSUAA_SERVICE:
                return await self._handle_xsuaa_service(request)
            elif request.service_type == SAPBTPServiceType.NOTIFICATION_SERVICE:
                return await self._handle_notification_service(request)
            else:
                raise ValueError(f"Unsupported service type: {request.service_type}")
                
        except Exception as e:
            logger.error(f"SAP BTP request failed: {e}")
            return A2ASAPBTPResponse(
                success=False,
                error=str(e),
                service_type=request.service_type,
                operation=request.operation
            )

    async def _handle_destination_service(self, request: A2ASAPBTPRequest) -> A2ASAPBTPResponse:
        """Handle Destination Service requests through A2A messaging"""
        try:
            if request.operation == "get_access_token":
                token_data = await self._get_destination_service_token_a2a(request.parameters)
                return A2ASAPBTPResponse(
                    success=True,
                    data=token_data,
                    service_type=request.service_type,
                    operation=request.operation
                )
                
            elif request.operation == "get_destination":
                destination_data = await self._get_destination_config_a2a(request.parameters)
                return A2ASAPBTPResponse(
                    success=True,
                    data=destination_data,
                    service_type=request.service_type,
                    operation=request.operation
                )
                
            elif request.operation == "list_destinations":
                destinations = await self._list_destinations_a2a(request.parameters)
                return A2ASAPBTPResponse(
                    success=True,
                    data={"destinations": destinations},
                    service_type=request.service_type,
                    operation=request.operation
                )
                
            else:
                raise ValueError(f"Unsupported Destination Service operation: {request.operation}")
                
        except Exception as e:
            raise Exception(f"Destination Service operation failed: {e}")

    async def _get_destination_service_token_a2a(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Get Destination Service access token through A2A messaging"""
        
        # Create A2A message for external service bridge agent
        bridge_message = {
            "service_type": "oauth2_token",
            "endpoint": params.get("token_url"),
            "method": "POST",
            "headers": params.get("headers", {}),
            "data": params.get("data", {}),
            "authentication": params.get("authentication", {})
        }
        
        # Send through A2A network to external service bridge
        response = await self.network_client.send_a2a_message(
            to_agent="external_service_bridge",
            message=bridge_message,
            message_type="SERVICE_REQUEST"
        )
        
        if response.get("success"):
            token_data = response.get("data", {})
            
            # Cache token with expiration
            cache_key = "destination_service_token"
            expires_in = token_data.get("expires_in", 3600)
            self.token_cache[cache_key] = {
                "access_token": token_data.get("access_token"),
                "expires_at": datetime.utcnow() + timedelta(seconds=expires_in - 60)
            }
            
            return {"access_token": token_data.get("access_token")}
        else:
            raise Exception(f"Token request failed: {response.get('error')}")

    async def _get_destination_config_a2a(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get destination configuration through A2A messaging"""
        
        destination_name = params.get("destination_name")
        user_token = params.get("user_token")
        
        # Check cache first
        cache_key = f"{destination_name}_{user_token or 'system'}"
        # Note: In production, implement proper cache with TTL
        
        # Get access token
        access_token = await self._get_cached_or_new_token()
        
        # Create A2A message for external service bridge
        bridge_message = {
            "service_type": "destination_config",
            "endpoint": params.get("service_url") + f"/destination-configuration/v1/destinations/{destination_name}",
            "method": "GET",
            "headers": {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        }
        
        if user_token:
            bridge_message["headers"]["X-User-Token"] = user_token
        
        # Send through A2A network
        response = await self.network_client.send_a2a_message(
            to_agent="external_service_bridge",
            message=bridge_message,
            message_type="SERVICE_REQUEST"
        )
        
        if response.get("success"):
            return response.get("data", {})
        else:
            raise Exception(f"Destination config request failed: {response.get('error')}")

    async def _list_destinations_a2a(self, params: Dict[str, Any]) -> List[str]:
        """List destinations through A2A messaging"""
        
        # Get access token
        access_token = await self._get_cached_or_new_token()
        
        # Create A2A message for external service bridge
        bridge_message = {
            "service_type": "destinations_list",
            "endpoint": params.get("service_url") + "/destination-configuration/v1/destinations",
            "method": "GET",
            "headers": {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        }
        
        # Send through A2A network
        response = await self.network_client.send_a2a_message(
            to_agent="external_service_bridge",
            message=bridge_message,
            message_type="SERVICE_REQUEST"
        )
        
        if response.get("success"):
            destinations_data = response.get("data", [])
            return [dest.get("Name", "") for dest in destinations_data]
        else:
            logger.error(f"List destinations failed: {response.get('error')}")
            return []

    async def _handle_xsuaa_service(self, request: A2ASAPBTPRequest) -> A2ASAPBTPResponse:
        """Handle XSUAA service requests through A2A messaging"""
        try:
            if request.operation == "validate_token":
                validation_result = await self._validate_jwt_token_a2a(request.parameters)
                return A2ASAPBTPResponse(
                    success=True,
                    data=validation_result,
                    service_type=request.service_type,
                    operation=request.operation
                )
            else:
                raise ValueError(f"Unsupported XSUAA operation: {request.operation}")
                
        except Exception as e:
            raise Exception(f"XSUAA service operation failed: {e}")

    async def _validate_jwt_token_a2a(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JWT token through A2A messaging"""
        
        # Create A2A message for JWT validation service
        validation_message = {
            "service_type": "jwt_validation",
            "token": params.get("token"),
            "xsuaa_config": params.get("xsuaa_config", {}),
            "validation_options": params.get("validation_options", {})
        }
        
        # Send through A2A network to JWT validation agent
        response = await self.network_client.send_a2a_message(
            to_agent="jwt_validation_agent",
            message=validation_message,
            message_type="VALIDATION_REQUEST"
        )
        
        if response.get("success"):
            return response.get("data", {})
        else:
            raise Exception(f"JWT validation failed: {response.get('error')}")

    async def _handle_notification_service(self, request: A2ASAPBTPRequest) -> A2ASAPBTPResponse:
        """Handle Notification service requests through A2A messaging"""
        try:
            if request.operation == "send_notification":
                result = await self._send_notification_a2a(request.parameters)
                return A2ASAPBTPResponse(
                    success=True,
                    data=result,
                    service_type=request.service_type,
                    operation=request.operation
                )
            else:
                raise ValueError(f"Unsupported Notification operation: {request.operation}")
                
        except Exception as e:
            raise Exception(f"Notification service operation failed: {e}")

    async def _send_notification_a2a(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification through A2A messaging"""
        
        # Create A2A message for notification service
        notification_message = {
            "service_type": "alert_notification",
            "notification": params.get("notification", {}),
            "recipients": params.get("recipients", []),
            "config": params.get("config", {})
        }
        
        # Send through A2A network to notification agent
        response = await self.network_client.send_a2a_message(
            to_agent="notification_agent",
            message=notification_message,
            message_type="NOTIFICATION_REQUEST"
        )
        
        if response.get("success"):
            return response.get("data", {})
        else:
            raise Exception(f"Notification sending failed: {response.get('error')}")

    async def _get_cached_or_new_token(self) -> str:
        """Get cached token or request new one"""
        cache_key = "destination_service_token"
        
        # Check cache
        if cache_key in self.token_cache:
            token_info = self.token_cache[cache_key]
            if datetime.utcnow() < token_info["expires_at"]:
                return token_info["access_token"]
        
        # Request new token
        token_data = await self._get_destination_service_token_a2a({
            "grant_type": "client_credentials"
        })
        
        return token_data.get("access_token", "")

    def configure_service(self, service_type: SAPBTPServiceType, config: Dict[str, Any]):
        """Configure SAP BTP service"""
        self.service_configs[service_type.value] = config
        logger.info(f"Configured {service_type.value} service")

    async def health_check(self) -> Dict[str, Any]:
        """A2A SAP BTP agent health check"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "capabilities": [cap.value for cap in self.capabilities],
            "services_configured": list(self.service_configs.keys()),
            "tokens_cached": len(self.token_cache),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global agent instance
_sap_btp_agent: Optional[A2ASAPBTPAgent] = None


def initialize_sap_btp_agent(agent_id: str = "sap_btp_agent") -> A2ASAPBTPAgent:
    """Initialize global SAP BTP agent"""
    global _sap_btp_agent
    
    if _sap_btp_agent is None:
        _sap_btp_agent = A2ASAPBTPAgent(agent_id=agent_id)
    
    return _sap_btp_agent


def get_sap_btp_agent() -> Optional[A2ASAPBTPAgent]:
    """Get the global SAP BTP agent"""
    return _sap_btp_agent


# Convenience functions for A2A SAP BTP operations
async def send_sap_btp_request(
    service_type: SAPBTPServiceType,
    operation: str,
    parameters: Dict[str, Any],
    from_agent: str
) -> A2ASAPBTPResponse:
    """Send SAP BTP request through A2A protocol"""
    
    agent = get_sap_btp_agent()
    if not agent:
        agent = initialize_sap_btp_agent()
    
    request = A2ASAPBTPRequest(
        service_type=service_type,
        operation=operation,
        parameters=parameters
    )
    
    return await agent._handle_sap_btp_request(request)
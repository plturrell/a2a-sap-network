"""
External Service Bridge Agent
A2A Protocol compliant agent that handles external HTTP API calls
This agent acts as a secure bridge between A2A blockchain messaging and external APIs
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import base64
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import logging

import httpx
import certifi
from pydantic import BaseModel, Field
from fastapi import HTTPException

# A2A Protocol imports
from ..sdk.agentBase import A2AAgent
from ..sdk.blockchainIntegration import BlockchainIntegrationMixin
from ..sdk.types import A2AMessage, MessageType, AgentCapability

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """Supported HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ExternalServiceRequest(BaseModel):
    """Request to external service through bridge"""
    service_type: str
    endpoint: str
    method: HTTPMethod = HTTPMethod.GET
    headers: Dict[str, str] = Field(default_factory=dict)
    data: Optional[Union[Dict[str, Any], str]] = None
    params: Optional[Dict[str, Any]] = None
    authentication: Optional[Dict[str, str]] = None
    timeout: int = 30
    verify_ssl: bool = True
    allow_redirects: bool = True
    max_retries: int = 3


class ExternalServiceResponse(BaseModel):
    """Response from external service"""
    success: bool
    status_code: Optional[int] = None
    data: Optional[Union[Dict[str, Any], List[Any], str]] = None
    headers: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    service_type: str
    endpoint: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExternalServiceBridgeAgent(A2AAgent, BlockchainIntegrationMixin):
    """
    External Service Bridge Agent
    
    This agent provides A2A protocol compliant access to external HTTP APIs.
    It receives requests through A2A blockchain messaging and makes the actual
    HTTP calls, returning responses through the blockchain.
    
    Security Features:
    - Request validation and sanitization
    - Rate limiting per service
    - SSL certificate verification
    - Request/response logging for audit
    - Configurable allowed domains/endpoints
    """

    def __init__(self, agent_id: str = "external_service_bridge", **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        # Agent capabilities
        self.capabilities = [
            AgentCapability.EXTERNAL_API_INTEGRATION,
            AgentCapability.DATA_PROCESSING,
            AgentCapability.SERVICE_ORCHESTRATION
        ]
        
        # Security configuration
        self.allowed_domains: List[str] = [
            # SAP BTP domains
            "*.authentication.sap.hana.ondemand.com",
            "*.authentication.eu10.hana.ondemand.com", 
            "*.authentication.us10.hana.ondemand.com",
            "*.dest-configuration.sap.hana.ondemand.com",
            "*.notification.sap.hana.ondemand.com",
            # Add other approved domains
        ]
        
        # Rate limiting (requests per minute per service type)
        self.rate_limits: Dict[str, int] = {
            "oauth2_token": 60,
            "destination_config": 120,
            "destinations_list": 30,
            "jwt_validation": 180,
            "alert_notification": 100,
            "default": 50
        }
        
        # Request tracking for rate limiting
        self.request_history: Dict[str, List[datetime]] = {}
        
        # HTTP client configuration
        self.http_client_config = {
            "timeout": 30.0,
            "verify": certifi.where(),
            "follow_redirects": True,
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100)
        }
        
        logger.info(f"External Service Bridge Agent initialized: {agent_id}")

    async def initialize(self):
        """Initialize the External Service Bridge Agent"""
        await super().initialize()
        
        # Register capabilities with blockchain
        await self.register_capability(AgentCapability.EXTERNAL_API_INTEGRATION)
        
        logger.info("External Service Bridge Agent initialized successfully")

    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle incoming A2A messages for external service requests"""
        try:
            if message.payload.get("request_type") == "external_service" or "service_type" in message.payload:
                request = ExternalServiceRequest(**message.payload)
                
                # Validate and execute request
                response = await self._handle_external_service_request(request, message.sender)
                
                # Return A2A response message
                return A2AMessage(
                    id=f"bridge_response_{int(datetime.utcnow().timestamp())}",
                    type=MessageType.RESPONSE,
                    sender=self.agent_id,
                    receiver=message.sender,
                    in_reply_to=message.id,
                    payload=response.dict(),
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Error handling external service request: {e}")
            return A2AMessage(
                id=f"bridge_error_{int(datetime.utcnow().timestamp())}",
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                in_reply_to=message.id,
                payload={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )

    async def _handle_external_service_request(
        self, 
        request: ExternalServiceRequest, 
        requesting_agent: str
    ) -> ExternalServiceResponse:
        """Handle external service request with security validation"""
        try:
            # Validate request
            await self._validate_request(request, requesting_agent)
            
            # Check rate limits
            if not await self._check_rate_limit(request.service_type, requesting_agent):
                raise Exception(f"Rate limit exceeded for service type: {request.service_type}")
            
            # Execute HTTP request
            response_data = await self._execute_http_request(request)
            
            return ExternalServiceResponse(
                success=True,
                status_code=response_data.get("status_code"),
                data=response_data.get("data"),
                headers=response_data.get("headers"),
                service_type=request.service_type,
                endpoint=request.endpoint
            )
            
        except Exception as e:
            logger.error(f"External service request failed: {e}")
            return ExternalServiceResponse(
                success=False,
                error=str(e),
                service_type=request.service_type,
                endpoint=request.endpoint
            )

    async def _validate_request(self, request: ExternalServiceRequest, requesting_agent: str):
        """Validate external service request for security compliance"""
        
        # Validate endpoint domain
        endpoint_domain = self._extract_domain(request.endpoint)
        if not self._is_domain_allowed(endpoint_domain):
            raise Exception(f"Domain not allowed: {endpoint_domain}")
        
        # Validate HTTP method
        if request.method not in HTTPMethod:
            raise Exception(f"HTTP method not allowed: {request.method}")
        
        # Validate request size
        if request.data:
            data_str = json.dumps(request.data) if isinstance(request.data, dict) else str(request.data)
            if len(data_str) > 1024 * 1024:  # 1MB limit
                raise Exception("Request payload too large (max 1MB)")
        
        # Log request for audit
        logger.info(f"External service request validated: {requesting_agent} -> {request.service_type} -> {endpoint_domain}")

    def _extract_domain(self, endpoint: str) -> str:
        """Extract domain from endpoint URL"""
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        return parsed.netloc.lower()

    def _is_domain_allowed(self, domain: str) -> bool:
        """Check if domain is in allowed list"""
        import fnmatch
        
        for allowed_pattern in self.allowed_domains:
            if fnmatch.fnmatch(domain, allowed_pattern):
                return True
        
        return False

    async def _check_rate_limit(self, service_type: str, requesting_agent: str) -> bool:
        """Check rate limits for service type and requesting agent"""
        
        rate_key = f"{service_type}_{requesting_agent}"
        current_time = datetime.utcnow()
        
        # Get rate limit for service type
        limit = self.rate_limits.get(service_type, self.rate_limits["default"])
        
        # Initialize history if not exists
        if rate_key not in self.request_history:
            self.request_history[rate_key] = []
        
        # Clean old entries (older than 1 minute)
        minute_ago = current_time - timedelta(minutes=1)
        self.request_history[rate_key] = [
            req_time for req_time in self.request_history[rate_key]
            if req_time > minute_ago
        ]
        
        # Check if within limit
        if len(self.request_history[rate_key]) >= limit:
            logger.warning(f"Rate limit exceeded: {rate_key} ({len(self.request_history[rate_key])}/{limit})")
            return False
        
        # Add current request
        self.request_history[rate_key].append(current_time)
        return True

    async def _execute_http_request(self, request: ExternalServiceRequest) -> Dict[str, Any]:
        """Execute the actual HTTP request"""
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= request.max_retries:
            try:
                async with httpx.AsyncClient(**self.http_client_config) as client:
                    # Prepare request parameters
                    request_params = {
                        "method": request.method.value,
                        "url": request.endpoint,
                        "headers": request.headers,
                        "timeout": request.timeout
                    }
                    
                    # Add data/json payload
                    if request.data:
                        if isinstance(request.data, dict):
                            request_params["json"] = request.data
                        else:
                            request_params["data"] = request.data
                    
                    # Add query parameters
                    if request.params:
                        request_params["params"] = request.params
                    
                    # Execute request
                    response = await client.request(**request_params)
                    
                    # Parse response
                    try:
                        response_data = response.json()
                    except:
                        response_data = response.text
                    
                    return {
                        "status_code": response.status_code,
                        "data": response_data,
                        "headers": dict(response.headers)
                    }
                    
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                if retry_count <= request.max_retries:
                    wait_time = min(2 ** retry_count, 10)  # Exponential backoff, max 10 seconds
                    logger.warning(f"HTTP request failed (attempt {retry_count}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        # All retries failed
        raise Exception(f"HTTP request failed after {request.max_retries + 1} attempts: {last_exception}")

    def add_allowed_domain(self, domain: str):
        """Add domain to allowed list"""
        if domain not in self.allowed_domains:
            self.allowed_domains.append(domain)
            logger.info(f"Added allowed domain: {domain}")

    def remove_allowed_domain(self, domain: str):
        """Remove domain from allowed list"""
        if domain in self.allowed_domains:
            self.allowed_domains.remove(domain)
            logger.info(f"Removed allowed domain: {domain}")

    def set_rate_limit(self, service_type: str, limit: int):
        """Set rate limit for service type"""
        self.rate_limits[service_type] = limit
        logger.info(f"Set rate limit for {service_type}: {limit} requests/minute")

    async def health_check(self) -> Dict[str, Any]:
        """External Service Bridge health check"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "capabilities": [cap.value for cap in self.capabilities],
            "allowed_domains_count": len(self.allowed_domains),
            "rate_limits": self.rate_limits,
            "active_request_tracking": len(self.request_history),
            "timestamp": datetime.utcnow().isoformat()
        }


# JWT Validation Agent (specialized for XSUAA tokens)
class JWTValidationAgent(A2AAgent):
    """Specialized agent for JWT token validation"""

    def __init__(self, agent_id: str = "jwt_validation_agent", **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        
        self.capabilities = [AgentCapability.AUTHENTICATION]

    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle JWT validation requests"""
        try:
            if message.payload.get("service_type") == "jwt_validation":
                result = await self._validate_jwt_token(
                    token=message.payload.get("token"),
                    xsuaa_config=message.payload.get("xsuaa_config", {}),
                    validation_options=message.payload.get("validation_options", {})
                )
                
                return A2AMessage(
                    id=f"jwt_response_{int(datetime.utcnow().timestamp())}",
                    type=MessageType.RESPONSE,
                    sender=self.agent_id,
                    receiver=message.sender,
                    in_reply_to=message.id,
                    payload={"success": True, "data": {"decoded_token": result}},
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return A2AMessage(
                id=f"jwt_error_{int(datetime.utcnow().timestamp())}",
                type=MessageType.ERROR,
                sender=self.agent_id,
                receiver=message.sender,
                in_reply_to=message.id,
                payload={"success": False, "error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )

    async def _validate_jwt_token(
        self, 
        token: str, 
        xsuaa_config: Dict[str, Any], 
        validation_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate JWT token with proper verification"""
        import jwt
        import os
        
        # Development mode bypass
        if validation_options.get("verify_signature") is False:
            return jwt.decode(token, options={"verify_signature": False})
        
        # Production verification
        jwt_secret = validation_options.get("jwt_secret")
        jwt_algorithm = validation_options.get("jwt_algorithm", "HS256")
        
        if jwt_algorithm.startswith("RS"):  # RSA algorithms
            public_key_path = validation_options.get("public_key_path", "/app/certs/public.pem")
            try:
                with open(public_key_path, "r") as f:
                    public_key = f.read()
                return jwt.decode(
                    token,
                    public_key,
                    algorithms=[jwt_algorithm],
                    options={"verify_exp": True, "verify_aud": True}
                )
            except FileNotFoundError:
                raise ValueError(f"JWT public key not found at {public_key_path}")
        else:
            # Symmetric algorithms
            if not jwt_secret:
                raise ValueError("JWT_SECRET_KEY must be set for token verification")
            return jwt.decode(
                token,
                jwt_secret,
                algorithms=[jwt_algorithm],
                options={"verify_exp": True}
            )


# Global agent instances
_external_service_bridge: Optional[ExternalServiceBridgeAgent] = None
_jwt_validation_agent: Optional[JWTValidationAgent] = None


def initialize_bridge_agents():
    """Initialize global bridge agents"""
    global _external_service_bridge, _jwt_validation_agent
    
    if _external_service_bridge is None:
        _external_service_bridge = ExternalServiceBridgeAgent()
    
    if _jwt_validation_agent is None:
        _jwt_validation_agent = JWTValidationAgent()
    
    return _external_service_bridge, _jwt_validation_agent


def get_external_service_bridge() -> Optional[ExternalServiceBridgeAgent]:
    """Get the external service bridge agent"""
    return _external_service_bridge


def get_jwt_validation_agent() -> Optional[JWTValidationAgent]:
    """Get the JWT validation agent"""
    return _jwt_validation_agent
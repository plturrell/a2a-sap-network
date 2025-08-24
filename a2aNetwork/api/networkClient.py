"""
A2A Network Client - Main interface for a2aAgents to access network services
"""

import asyncio
# A2A Protocol: Use blockchain messaging instead of httpx
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from .registryApi import RegistryAPI
from .trustApi import TrustAPI
from .sdkApi import SdkAPI

logger = logging.getLogger(__name__)


class NetworkClient:
    """
    Main client for accessing A2A Network services
    Provides unified interface to registry, trust, and SDK services
    """
    
    def __init__(self, 
                 registry_url: str = "http://localhost:9000",
                 trust_service_url: str = "http://localhost:9001",
                 timeout: int = 30):
        """
        Initialize A2A Network Client
        
        Args:
            registry_url: URL of the registry service
            trust_service_url: URL of the trust service
            timeout: Request timeout in seconds
        """
        self.registry_url = registry_url
        self.trust_service_url = trust_service_url
        self.timeout = timeout
        
        # Initialize service APIs
        self.registry = RegistryAPI(registry_url, timeout)
        self.trust = TrustAPI(trust_service_url, timeout)
        self.sdk = SdkAPI()
        
        # HTTP client for direct network calls
        self.http_client = None
        
        logger.info(f"Initialized A2A Network Client")
        logger.info(f"  Registry: {registry_url}")
        logger.info(f"  Trust Service: {trust_service_url}")
    
    async def initialize(self) -> bool:
        """Initialize the network client and verify connectivity"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            
            # Initialize service APIs
            await self.registry.initialize()
            await self.trust.initialize()
            
            # Verify connectivity
            registry_status = await self.registry.health_check()
            trust_status = await self.trust.health_check()
            
            if registry_status and trust_status:
                logger.info("✅ A2A Network Client successfully connected to all services")
                return True
            else:
                logger.warning("⚠️  Some A2A Network services are unavailable")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize A2A Network Client: {e}")
            return False
    
    async def shutdown(self):
        """Clean shutdown of the network client"""
        try:
            if self.http_client:
                await self.http_client.aclose()
                
            await self.registry.shutdown()
            await self.trust.shutdown()
            
            logger.info("A2A Network Client shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during network client shutdown: {e}")
    
    # Registry Operations
    async def register_agent(self, agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """Register an agent with the A2A Registry"""
        return await self.registry.register_agent(agent_card)
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information from registry"""
        return await self.registry.get_agent(agent_id)
    
    async def find_agents(self, capabilities: List[str] = None, 
                         domain: str = None) -> List[Dict[str, Any]]:
        """Find agents by capabilities or domain"""
        return await self.registry.find_agents(capabilities, domain)
    
    async def register_data_product(self, dublin_core_metadata: Dict[str, Any],
                                  ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register a data product with the registry"""
        return await self.registry.register_data_product(dublin_core_metadata, ord_descriptor)
    
    # Trust Operations
    async def initialize_agent_trust(self, agent_id: str, 
                                   base_url: str) -> Dict[str, Any]:
        """Initialize trust identity for an agent"""
        return await self.trust.initialize_agent_trust(agent_id, base_url)
    
    async def verify_agent_trust(self, agent_id: str, 
                               signature: str, 
                               message: str) -> bool:
        """Verify trust signature for an agent"""
        return await self.trust.verify_agent_trust(agent_id, signature, message)
    
    async def get_trust_score(self, agent_id: str) -> float:
        """Get trust score for an agent"""
        return await self.trust.get_trust_score(agent_id)
    
    # Network Messaging
    async def send_message(self, 
                          from_agent: str,
                          to_agent: str, 
                          message: Dict[str, Any],
                          trust_verify: bool = True) -> Dict[str, Any]:
        """
        Send A2A message between agents through the network
        
        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID  
            message: A2A message payload
            trust_verify: Whether to verify trust before sending
            
        Returns:
            Response from recipient agent
        """
        try:
            # Get recipient agent info
            recipient = await self.get_agent(to_agent)
            if not recipient:
                raise Exception(f"Agent {to_agent} not found in registry")
            
            # Verify trust if requested
            if trust_verify:
                trust_score = await self.get_trust_score(to_agent)
                if trust_score < 0.5:  # Configurable threshold
                    logger.warning(f"Low trust score for agent {to_agent}: {trust_score}")
            
            # Send message to recipient
            recipient_url = recipient.get("base_url", "")
            if not recipient_url:
                raise Exception(f"No base URL found for agent {to_agent}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{recipient_url}/api/v1/message",
                    json={
                        "from": from_agent,
                        "to": to_agent,
                        "message": message,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Message delivery failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Network Status
    async def get_network_status(self) -> Dict[str, Any]:
        """Get overall network status"""
        try:
            registry_status = await self.registry.get_status()
            trust_status = await self.trust.get_status()
            
            # Get network statistics
            total_agents = await self.registry.get_agent_count()
            active_agents = await self.registry.get_active_agent_count()
            
            return {
                "network_healthy": registry_status["healthy"] and trust_status["healthy"],
                "services": {
                    "registry": registry_status,
                    "trust": trust_status
                },
                "statistics": {
                    "total_agents": total_agents,
                    "active_agents": active_agents,
                    "network_uptime": registry_status.get("uptime", 0)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {
                "network_healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Convenience function for creating network client
def create_network_client(registry_url: str = "http://localhost:9000",
                         trust_service_url: str = "http://localhost:9001",
                         timeout: int = 30) -> NetworkClient:
    """Create and return a configured NetworkClient instance"""
    return NetworkClient(registry_url, trust_service_url, timeout)
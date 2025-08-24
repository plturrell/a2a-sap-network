"""
Registry API - Interface for A2A Registry services
"""

# A2A Protocol: Use blockchain messaging instead of httpx
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RegistryAPI:
    """API interface for A2A Registry operations"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.http_client = None
    
    async def initialize(self):
        """Initialize the registry API client"""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
            logger.info(f"Registry API initialized: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Registry API: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the registry API client"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def health_check(self) -> bool:
        """Check if registry service is healthy"""
        try:
            if not self.http_client:
                await self.initialize()
            
            response = await self.http_client.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Registry health check failed: {e}")
            return False
    
    async def register_agent(self, agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register an agent with the registry
        
        Args:
            agent_card: Agent registration data
            
        Returns:
            Registration result
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/agents/register",
                json=agent_card
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Agent registered successfully: {agent_card.get('agent_id')}")
                return result
            else:
                raise Exception(f"Registration failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information by ID"""
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(f"{self.base_url}/api/v1/agents/{agent_id}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                raise Exception(f"Failed to get agent: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}")
            return None
    
    async def find_agents(self, capabilities: List[str] = None, 
                         domain: str = None) -> List[Dict[str, Any]]:
        """Find agents by capabilities or domain"""
        try:
            if not self.http_client:
                await self.initialize()
            
            params = {}
            if capabilities:
                params['capabilities'] = ','.join(capabilities)
            if domain:
                params['domain'] = domain
            
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/agents/search",
                params=params
            )
            
            if response.status_code == 200:
                return response.json().get('agents', [])
            else:
                raise Exception(f"Search failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error finding agents: {e}")
            return []
    
    async def register_data_product(self, dublin_core_metadata: Dict[str, Any],
                                  ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register a data product with ORD compliance"""
        try:
            if not self.http_client:
                await self.initialize()
            
            payload = {
                "dublin_core": dublin_core_metadata,
                "ord_descriptor": ord_descriptor,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/ord/register",
                json=payload
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Data product registered: {dublin_core_metadata.get('identifier')}")
                return result
            else:
                raise Exception(f"Data product registration failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error registering data product: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get registry service status"""
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(f"{self.base_url}/api/v1/status")
            
            if response.status_code == 200:
                status = response.json()
                status["healthy"] = True
                return status
            else:
                return {"healthy": False, "error": f"Status check failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting registry status: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def get_agent_count(self) -> int:
        """Get total number of registered agents"""
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(f"{self.base_url}/api/v1/agents/count")
            
            if response.status_code == 200:
                return response.json().get("count", 0)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error getting agent count: {e}")
            return 0
    
    async def get_active_agent_count(self) -> int:
        """Get number of active agents"""
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(f"{self.base_url}/api/v1/agents/count/active")
            
            if response.status_code == 200:
                return response.json().get("count", 0)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error getting active agent count: {e}")
            return 0
"""
Trust API - Interface for A2A Trust System services
"""

# A2A Protocol: Use blockchain messaging instead of httpx
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrustAPI:
    """API interface for A2A Trust System operations"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.http_client = None
    
    async def initialize(self):
        """Initialize the trust API client"""
        try:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
            logger.info(f"Trust API initialized: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Trust API: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the trust API client"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def health_check(self) -> bool:
        """Check if trust service is healthy"""
        try:
            if not self.http_client:
                await self.initialize()
            
            response = await self.http_client.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Trust service health check failed: {e}")
            return False
    
    async def initialize_agent_trust(self, agent_id: str, base_url: str) -> Dict[str, Any]:
        """
        Initialize trust identity for an agent
        
        Args:
            agent_id: Unique identifier for the agent
            base_url: Base URL where the agent is accessible
            
        Returns:
            Trust identity information including keys and addresses
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            payload = {
                "agent_id": agent_id,
                "base_url": base_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/trust/initialize",
                json=payload
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Trust identity initialized for agent: {agent_id}")
                return result
            else:
                raise Exception(f"Trust initialization failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to initialize trust for agent {agent_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def verify_agent_trust(self, agent_id: str, signature: str, message: str) -> bool:
        """
        Verify trust signature for an agent
        
        Args:
            agent_id: Agent identifier
            signature: Digital signature to verify
            message: Original message that was signed
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            payload = {
                "agent_id": agent_id,
                "signature": signature,
                "message": message
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/trust/verify",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("verified", False)
            else:
                logger.warning(f"Trust verification failed for agent {agent_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying trust for agent {agent_id}: {e}")
            return False
    
    async def get_trust_score(self, agent_id: str) -> float:
        """
        Get trust score for an agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/trust/score/{agent_id}"
            )
            
            if response.status_code == 200:
                result = response.json()
                return float(result.get("trust_score", 0.0))
            elif response.status_code == 404:
                logger.info(f"No trust score found for agent {agent_id}")
                return 0.0
            else:
                raise Exception(f"Failed to get trust score: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting trust score for agent {agent_id}: {e}")
            return 0.0
    
    async def record_trust_event(self, agent_id: str, event_type: str, 
                                details: Dict[str, Any] = None) -> bool:
        """
        Record a trust-related event for an agent
        
        Args:
            agent_id: Agent identifier
            event_type: Type of trust event (success, failure, delegation, etc.)
            details: Additional event details
            
        Returns:
            True if event was recorded successfully
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            payload = {
                "agent_id": agent_id,
                "event_type": event_type,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/trust/events",
                json=payload
            )
            
            if response.status_code == 201:
                logger.debug(f"Trust event recorded for agent {agent_id}: {event_type}")
                return True
            else:
                logger.warning(f"Failed to record trust event: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error recording trust event for agent {agent_id}: {e}")
            return False
    
    async def get_trusted_agents(self, agent_id: str) -> List[str]:
        """
        Get list of agents trusted by the specified agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of trusted agent IDs
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/trust/trusted/{agent_id}"
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("trusted_agents", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting trusted agents for {agent_id}: {e}")
            return []
    
    async def establish_trust_relationship(self, from_agent: str, to_agent: str) -> bool:
        """
        Establish trust relationship between two agents
        
        Args:
            from_agent: Trusting agent ID
            to_agent: Trusted agent ID
            
        Returns:
            True if relationship established successfully
        """
        try:
            if not self.http_client:
                await self.initialize()
                
            payload = {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.base_url}/api/v1/trust/establish",
                json=payload
            )
            
            if response.status_code == 201:
                logger.info(f"Trust relationship established: {from_agent} -> {to_agent}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error establishing trust relationship: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get trust service status"""
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
            logger.error(f"Error getting trust service status: {e}")
            return {"healthy": False, "error": str(e)}
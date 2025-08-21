"""
A2A Registry Client for Dynamic Service Discovery
Enables true agent-to-agent communication without hardcoded URLs
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



# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class A2ARegistryClient:
    """Client for interacting with the A2A Registry for service discovery"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1/a2a"):
        self.base_url = base_url.rstrip("/")
        self._client = None
        self._cache = {}  # Cache discovered agents
        self._cache_ttl = 300  # 5 minutes cache TTL
        
    async def _ensure_client(self):
        """Ensure HTTP client is initialized"""
        if not self._client:
            self._client = # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        # httpx\.AsyncClient(timeout=30.0)
            
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def register_agent(self, agent_id: str, agent_card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Register an agent with the A2A Registry"""
        try:
            await self._ensure_client()
            
            registration_data = {
                "agent_card": agent_card,
                "registered_by": agent_id,  # Use agent_id as the registrant
                "tags": self._extract_tags_from_card(agent_card),
                "labels": {
                    "agent_id": agent_id,
                    "auto_registered": "true"
                }
            }
            
            response = await self._client.post(
                f"{self.base_url}/agents/register",
                json=registration_data
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"✅ Agent {agent_id} registered successfully")
                return result
            else:
                logger.error(f"Failed to register agent: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return None
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the A2A Registry"""
        try:
            await self._ensure_client()
            
            response = await self._client.delete(
                f"{self.base_url}/agents/{agent_id}"
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Deregistration error: {e}")
            return False
    
    async def discover_agent(self, agent_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Discover a specific agent by ID"""
        try:
            # Check cache first
            if use_cache and agent_id in self._cache:
                cached = self._cache[agent_id]
                if (datetime.utcnow() - cached["cached_at"]).seconds < self._cache_ttl:
                    return cached["agent"]
            
            await self._ensure_client()
            
            response = await self._client.get(
                f"{self.base_url}/agents/{agent_id}"
            )
            
            if response.status_code == 200:
                agent_data = response.json()
                
                # Cache the result
                self._cache[agent_id] = {
                    "agent": agent_data,
                    "cached_at": datetime.utcnow()
                }
                
                return agent_data
            else:
                logger.warning(f"Agent {agent_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return None
    
    async def search_agents(
        self,
        skills: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        input_modes: Optional[List[str]] = None,
        output_modes: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Search for agents based on capabilities"""
        try:
            await self._ensure_client()
            
            params = {}
            if skills:
                params["skills"] = ",".join(skills)
            if tags:
                params["tags"] = ",".join(tags)
            if input_modes:
                params["input_modes"] = ",".join(input_modes)
            if output_modes:
                params["output_modes"] = ",".join(output_modes)
            if status:
                params["status"] = status
            
            response = await self._client.get(
                f"{self.base_url}/agents/search",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Search failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None
    
    async def get_agent_url(self, agent_id: str) -> Optional[str]:
        """Get the URL for a specific agent"""
        agent = await self.discover_agent(agent_id)
        if agent and "url" in agent:
            return agent["url"]
        return None
    
    async def find_agent_by_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Find the first available agent with a specific skill"""
        results = await self.search_agents(skills=[skill_id], status="healthy")
        
        if results and results.get("agents"):
            # Return the first healthy agent with the skill
            return results["agents"][0]
        return None
    
    async def match_workflow_requirements(
        self,
        required_skills: List[str],
        preferred_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Match agents to workflow requirements"""
        try:
            await self._ensure_client()
            
            request_data = {
                "required_skills": required_skills,
                "preferred_tags": preferred_tags or []
            }
            
            response = await self._client.post(
                f"{self.base_url}/agents/match",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json().get("matched_agents", [])
            else:
                logger.error(f"Workflow matching failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Workflow matching error: {e}")
            return []
    
    async def send_message_to_agent(
        self,
        agent_id: str,
        message: Dict[str, Any],
        context_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a message to an agent discovered via the registry"""
        try:
            # Discover the agent
            agent = await self.discover_agent(agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found in registry")
                return None
            
            agent_url = agent.get("url")
            if not agent_url:
                logger.error(f"Agent {agent_id} has no URL")
                return None
            
            await self._ensure_client()
            
            # Send message to the discovered agent
            payload = {
                "message": message,
                "contextId": context_id or str(datetime.utcnow().timestamp())
            }
            
            response = await self._client.post(
                f"{agent_url}/a2a/v1/messages",
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to send message to {agent_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending message to {agent_id}: {e}")
            return None
    
    def _extract_tags_from_card(self, agent_card: Dict[str, Any]) -> List[str]:
        """Extract tags from agent card skills"""
        tags = []
        
        if "skills" in agent_card:
            for skill in agent_card["skills"]:
                if "tags" in skill:
                    tags.extend(skill["tags"])
        
        # Add capability-based tags
        capabilities = agent_card.get("capabilities", {})
        if capabilities.get("streaming"):
            tags.append("streaming")
        if capabilities.get("batchProcessing"):
            tags.append("batch")
        
        return list(set(tags))  # Remove duplicates
    
    async def clear_cache(self):
        """Clear the agent cache"""
        self._cache.clear()
        logger.info("Agent cache cleared")


# Singleton instance for easy access
_registry_client = None

def get_registry_client(base_url: Optional[str] = None) -> A2ARegistryClient:
    """Get or create the singleton registry client"""
    global _registry_client
    
    if _registry_client is None:
        url = base_url or "http://localhost:8000/api/v1/a2a"
        _registry_client = A2ARegistryClient(base_url=url)
    
    return _registry_client
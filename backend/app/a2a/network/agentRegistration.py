"""
Agent Registration Service - Automatic registration of agents with network
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .networkConnector import get_network_connector
from ..storage import get_distributed_storage

logger = logging.getLogger(__name__)


class AgentRegistrationService:
    """
    Service to handle automatic agent registration with a2aNetwork
    Integrates into agent lifecycle for seamless network connectivity
    """
    
    def __init__(self):
        self.registered_agents = {}
        self.registration_tasks = {}
        self.network_connector = None
        self.distributed_storage = None
    
    async def initialize(self):
        """Initialize the registration service"""
        try:
            self.network_connector = get_network_connector()
            await self.network_connector.initialize()
            
            # Initialize distributed storage
            self.distributed_storage = await get_distributed_storage()
            
            logger.info("AgentRegistrationService initialized with distributed storage")
        except Exception as e:
            logger.error(f"Failed to initialize AgentRegistrationService: {e}")
    
    async def register_agent_on_startup(self, agent_instance) -> Dict[str, Any]:
        """
        Register agent automatically during startup
        Called from agent initialization
        
        Args:
            agent_instance: A2AAgentBase instance
            
        Returns:
            Registration result
        """
        try:
            if not self.network_connector:
                await self.initialize()
            
            logger.info(f"Auto-registering agent: {agent_instance.agent_id}")
            
            # Register with network
            result = await self.network_connector.register_agent(agent_instance)
            
            if result.get("success", False):
                # Store registration info locally
                self.registered_agents[agent_instance.agent_id] = {
                    "agent": agent_instance,
                    "registration_result": result,
                    "registered_at": datetime.utcnow().isoformat(),
                    "status": "active"
                }
                
                # Store in distributed storage
                agent_data = {
                    "agent_id": agent_instance.agent_id,
                    "name": getattr(agent_instance, 'name', agent_instance.agent_id),
                    "description": getattr(agent_instance, 'description', ''),
                    "capabilities": getattr(agent_instance, 'capabilities', []),
                    "endpoint": getattr(agent_instance, 'endpoint', ''),
                    "status": "active",
                    "registered_at": datetime.utcnow().isoformat(),
                    "registration_result": result,
                    "public_key": getattr(agent_instance, 'public_key_pem', None)
                }
                
                await self.distributed_storage.register_agent(agent_instance.agent_id, agent_data)
                
                logger.info(f"✅ Agent {agent_instance.agent_id} auto-registered successfully")
                
                # Start periodic health check
                self._start_health_check(agent_instance.agent_id)
                
                return result
            else:
                logger.warning(f"⚠️  Agent {agent_instance.agent_id} registration failed: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Auto-registration failed for {agent_instance.agent_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_agent_capabilities(self, agent_id: str, new_capabilities: List[str]) -> bool:
        """
        Update agent capabilities in registry
        
        Args:
            agent_id: Agent identifier
            new_capabilities: Updated list of capabilities
            
        Returns:
            True if update successful
        """
        try:
            if agent_id not in self.registered_agents:
                logger.warning(f"Agent {agent_id} not found in registered agents")
                return False
            
            agent_info = self.registered_agents[agent_id]
            agent_instance = agent_info["agent"]
            
            # Update agent capabilities
            # Note: This would typically update the agent's skill definitions
            logger.info(f"Updating capabilities for agent {agent_id}: {new_capabilities}")
            
            # Re-register with updated info
            result = await self.network_connector.register_agent(agent_instance)
            
            if result.get("success", False):
                agent_info["registration_result"] = result
                agent_info["last_updated"] = datetime.utcnow().isoformat()
                
                # Update in distributed storage
                agent_data = await self.distributed_storage.get_agent(agent_id)
                if agent_data:
                    agent_data["capabilities"] = new_capabilities
                    agent_data["last_updated"] = datetime.utcnow().isoformat()
                    await self.distributed_storage.register_agent(agent_id, agent_data)
                
                logger.info(f"✅ Agent {agent_id} capabilities updated")
                return True
            else:
                logger.error(f"Failed to update capabilities for {agent_id}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating capabilities for {agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister agent from network
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if deregistration successful
        """
        try:
            if agent_id in self.registered_agents:
                # Cancel health check task
                if agent_id in self.registration_tasks:
                    self.registration_tasks[agent_id].cancel()
                    del self.registration_tasks[agent_id]
                
                # Mark as inactive
                self.registered_agents[agent_id]["status"] = "inactive"
                self.registered_agents[agent_id]["deregistered_at"] = datetime.utcnow().isoformat()
                
                # Remove from distributed storage
                await self.distributed_storage.deregister_agent(agent_id)
                
                logger.info(f"✅ Agent {agent_id} deregistered")
                return True
            else:
                logger.warning(f"Agent {agent_id} not found for deregistration")
                return False
                
        except Exception as e:
            logger.error(f"Error deregistering agent {agent_id}: {e}")
            return False
    
    def _start_health_check(self, agent_id: str):
        """Start periodic health check for registered agent"""
        async def health_check_loop():
            while agent_id in self.registered_agents and \
                  self.registered_agents[agent_id]["status"] == "active":
                try:
                    # Perform health check
                    await self._perform_health_check(agent_id)
                    
                    # Wait 60 seconds before next check
                    await asyncio.sleep(60)
                    
                except asyncio.CancelledError:
                    logger.info(f"Health check cancelled for agent {agent_id}")
                    break
                except Exception as e:
                    logger.error(f"Health check error for agent {agent_id}: {e}")
                    await asyncio.sleep(60)  # Continue checking even after errors
        
        # Start health check task
        task = asyncio.create_task(health_check_loop())
        self.registration_tasks[agent_id] = task
    
    async def _perform_health_check(self, agent_id: str):
        """Perform health check for registered agent"""
        try:
            agent_info = self.registered_agents.get(agent_id)
            if not agent_info:
                return
            
            agent_instance = agent_info["agent"]
            
            # Check if agent is still responsive
            # This is a simple check - in practice you might ping the agent's health endpoint
            if hasattr(agent_instance, 'base_url') and agent_instance.base_url:
                # Agent appears healthy
                agent_info["last_health_check"] = datetime.utcnow().isoformat()
                logger.debug(f"Health check passed for agent {agent_id}")
            else:
                logger.warning(f"Health check failed for agent {agent_id}")
                
        except Exception as e:
            logger.error(f"Health check error for agent {agent_id}: {e}")
    
    async def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents"""
        return self.registered_agents.copy()
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get network connectivity status"""
        if not self.network_connector:
            await self.initialize()
        
        return await self.network_connector.get_network_status()
    
    async def shutdown(self):
        """Clean shutdown of registration service"""
        try:
            # Cancel all health check tasks
            for task in self.registration_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.registration_tasks:
                await asyncio.gather(*self.registration_tasks.values(), return_exceptions=True)
            
            self.registration_tasks.clear()
            
            # Shutdown network connector
            if self.network_connector:
                await self.network_connector.shutdown()
            
            logger.info("AgentRegistrationService shutdown complete")
            
        except Exception as e:
            logger.error(f"AgentRegistrationService shutdown error: {e}")


# Global registration service instance
_registration_service = None

def get_registration_service() -> AgentRegistrationService:
    """Get global registration service instance"""
    global _registration_service
    
    if _registration_service is None:
        _registration_service = AgentRegistrationService()
    
    return _registration_service
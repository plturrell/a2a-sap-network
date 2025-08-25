"""
Network Connector - Main interface for a2aAgents to connect to a2aNetwork
"""

import sys
import os
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global network connector instance
_network_connector = None


class NetworkConnector:
    """
    Main connector for a2aAgents to access a2aNetwork services
    Handles automatic failover between network and local services
    """

    def __init__(self,
                 a2a_network_path: Optional[str] = None,
                 registry_url: Optional[str] = None,
                 trust_service_url: Optional[str] = None):
        """
        Initialize network connector

        Args:
            a2a_network_path: Path to a2aNetwork directory (defaults to config)
            registry_url: URL of registry service
            trust_service_url: URL of trust service
        """
        self.a2a_network_path = a2a_network_path or settings.A2A_NETWORK_PATH
        self.registry_url = registry_url or settings.REGISTRY_URL
        self.trust_service_url = trust_service_url or settings.TRUST_SERVICE_URL

        # Network client (will be initialized on first use)
        self._network_client = None
        self._network_available = False

        # Initialization status
        self._initialized = False

        logger.info("NetworkConnector created")
        logger.info("  a2aNetwork path: %s", a2a_network_path)
        logger.info("  Registry URL: %s", registry_url)
        logger.info("  Trust Service URL: %s", trust_service_url)

    async def initialize(self) -> bool:
        """Initialize network connectivity"""
        if self._initialized:
            return self._network_available

        try:
            # Try to import and initialize a2aNetwork client
            await self._initialize_network_client()
            self._initialized = True

            if self._network_available:
                logger.info("✅ NetworkConnector: Connected to a2aNetwork services")
            else:
                logger.info("⚠️  NetworkConnector: Running in local-only mode")

            return self._network_available

        except (ImportError, RuntimeError) as e:
            logger.error("NetworkConnector initialization failed: %s", e)
            self._network_available = False
            self._initialized = True
            return False

    async def _initialize_network_client(self):
        """Try to initialize the network client"""
        try:
            # Add a2aNetwork to Python path
            if self.a2a_network_path not in sys.path:
                sys.path.insert(0, self.a2a_network_path)

            # Try to import network client
            from api.networkClient import create_network_client

            # Create and initialize network client
            self._network_client = create_network_client(
                registry_url=self.registry_url,
                trust_service_url=self.trust_service_url
            )

            # Try to initialize (this will test connectivity)
            connected = await self._network_client.initialize()

            if connected:
                self._network_available = True
                logger.info("✅ a2aNetwork services connected successfully")
            else:
                self._network_available = False
                logger.warning("⚠️  a2aNetwork services unavailable, using local fallback")

        except ImportError as e:
            logger.warning("a2aNetwork not available: %s", e)
            self._network_available = False
        except RuntimeError as e:
            logger.error("Network client initialization failed: %s", e)
            self._network_available = False

    async def register_agent(self, agent_instance) -> Dict[str, Any]:
        """
        Register agent with the network (if available) or store locally

        Args:
            agent_instance: A2AAgentBase instance

        Returns:
            Registration result
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self._network_available and self._network_client:
                # Use network registration
                logger.info("Registering agent %s with a2aNetwork", agent_instance.agent_id)

                # Create agent card using network SDK
                agent_card = self._network_client.sdk.create_agent_card_from_base(agent_instance)

                # Register with network registry
                result = await self._network_client.register_agent(agent_card)

                if result.get("success", False):
                    logger.info("✅ Agent %s registered with a2aNetwork", agent_instance.agent_id)
                    return result
                else:
                    raise RuntimeError(f"Network registration failed: {result.get('error', 'Unknown error')}")
            else:
                # Fallback to local registration
                logger.info("Registering agent %s locally (network unavailable)", agent_instance.agent_id)
                return await self._register_agent_locally(agent_instance)

        except RuntimeError as e:
            logger.error("Agent registration failed: %s", e)
            # Fallback to local registration
            return await self._register_agent_locally(agent_instance)

    async def _register_agent_locally(self, agent_instance) -> Dict[str, Any]:
        """Register agent locally when network is unavailable"""
        try:
            # Create local agent record
            agent_record = {
                "agent_id": agent_instance.agent_id,
                "name": agent_instance.name,
                "description": agent_instance.description,
                "version": agent_instance.version,
                "base_url": agent_instance.base_url,
                "capabilities": list(agent_instance.skills.keys()),
                "status": "active",
                "registered_at": datetime.utcnow().isoformat(),
                "registration_type": "local"
            }

            # Store in local registry file
            from ..config.storageConfig import get_registry_cache_path
            local_registry_path = str(get_registry_cache_path() / "a2a_local_registry.json")

            import json
            local_registry = {}

            # Load existing registry
            if os.path.exists(local_registry_path):
                with open(local_registry_path, 'r', encoding="utf-8") as f:
                    local_registry = json.load(f)

            # Add agent
            local_registry[agent_instance.agent_id] = agent_record

            # Save registry
            with open(local_registry_path, 'w', encoding="utf-8") as f:
                json.dump(local_registry, f, indent=2)

            logger.info("✅ Agent %s registered locally", agent_instance.agent_id)

            return {
                "success": True,
                "agent_id": agent_instance.agent_id,
                "registration_type": "local",
                "registry_path": local_registry_path
            }

        except (IOError, TypeError) as e:
            logger.error("Local agent registration failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def find_agents(self, capabilities: List[str] = None,
                         domain: str = None) -> List[Dict[str, Any]]:
        """Find agents by capabilities or domain"""
        if not self._initialized:
            await self.initialize()

        try:
            if self._network_available and self._network_client:
                # Use network search
                return await self._network_client.find_agents(capabilities, domain)
            else:
                # Search local registry
                return await self._find_agents_locally(capabilities, domain)

        except RuntimeError as e:
            logger.error("Agent search failed: %s", e)
            return []

    async def _find_agents_locally(self, capabilities: List[str] = None,
                                  domain: str = None) -> List[Dict[str, Any]]:
        """Search local agent registry"""
        try:
            from ..config.storageConfig import get_registry_cache_path
            local_registry_path = str(get_registry_cache_path() / "a2a_local_registry.json")

            if not os.path.exists(local_registry_path):
                return []

            import json
            with open(local_registry_path, 'r', encoding="utf-8") as f:
                local_registry = json.load(f)

            agents = []
            for agent_data in local_registry.values():
                # Filter by capabilities if specified
                if capabilities:
                    agent_capabilities = set(agent_data.get("capabilities", []))
                    required_capabilities = set(capabilities)
                    if not required_capabilities.issubset(agent_capabilities):
                        continue

                # Filter by domain if specified (basic implementation)
                if domain:
                    agent_name = agent_data.get("name", "").lower()
                    if domain.lower() not in agent_name:
                        continue

                agents.append(agent_data)

            return agents

        except (IOError, TypeError) as e:
            logger.error("Local agent search failed: %s", e)
            return []

    async def send_message(self, from_agent: str, to_agent: str,
                          message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message between agents through network"""
        if not self._initialized:
            await self.initialize()

        try:
            if self._network_available and self._network_client:
                # Use network messaging
                return await self._network_client.send_message(from_agent, to_agent, message)
            else:
                # Local messaging fallback
                return await self._send_message_locally(from_agent, to_agent, message)

        except RuntimeError as e:
            logger.error("Message sending failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def _send_message_locally(self, from_agent: str, to_agent: str,
                                   message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message locally when network unavailable"""
        # For local mode, we'll just log the message
        logger.info("LOCAL MESSAGE: %s -> %s", from_agent, to_agent)
        logger.info("Message content: %s", message)

        return {
            "success": True,
            "delivery_type": "local",
            "message": "Message logged locally (network unavailable)"
        }

    async def register_data_product(self, dublin_core_metadata: Dict[str, Any],
                                   ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register data product with registry"""
        if not self._initialized:
            await self.initialize()

        try:
            if self._network_available and self._network_client:
                # Use network registry
                return await self._network_client.register_data_product(
                    dublin_core_metadata, ord_descriptor
                )
            else:
                # Local data product registration
                return await self._register_data_product_locally(
                    dublin_core_metadata, ord_descriptor
                )

        except RuntimeError as e:
            logger.error("Data product registration failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def _register_data_product_locally(self, dublin_core_metadata: Dict[str, Any],
                                            ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register data product locally"""
        try:
            # Store in local data products file
            from ..config.storageConfig import get_registry_cache_path
            local_dp_path = str(get_registry_cache_path() / "a2a_local_data_products.json")

            import json


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            data_products = {}

            # Load existing data products
            if os.path.exists(local_dp_path):
                with open(local_dp_path, 'r', encoding="utf-8") as f:
                    data_products = json.load(f)

            # Create data product record
            dp_id = dublin_core_metadata.get("identifier", f"dp-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

            data_product_record = {
                "id": dp_id,
                "dublin_core": dublin_core_metadata,
                "ord_descriptor": ord_descriptor,
                "registered_at": datetime.utcnow().isoformat(),
                "registration_type": "local"
            }

            # Add data product
            data_products[dp_id] = data_product_record

            # Save data products
            with open(local_dp_path, 'w', encoding="utf-8") as f:
                json.dump(data_products, f, indent=2)

            logger.info("✅ Data product %s registered locally", dp_id)

            return {
                "success": True,
                "data_product_id": dp_id,
                "registration_type": "local"
            }

        except (IOError, TypeError) as e:
            logger.error("Local data product registration failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }

    async def get_network_status(self) -> Dict[str, Any]:
        """Get overall network connectivity status"""
        if not self._initialized:
            await self.initialize()

        status = {
            "initialized": self._initialized,
            "network_available": self._network_available,
            "registry_url": self.registry_url,
            "trust_service_url": self.trust_service_url,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self._network_available and self._network_client:
            try:
                network_status = await self._network_client.get_network_status()
                status.update(network_status)
            except RuntimeError as e:
                logger.error("Failed to get network status: %s", e)
                status["network_error"] = str(e)

        return status

    async def shutdown(self):
        """Clean shutdown of network connector"""
        try:
            if self._network_client:
                await self._network_client.shutdown()

            logger.info("NetworkConnector shutdown complete")

        except RuntimeError as e:
            logger.error("NetworkConnector shutdown error: %s", e)


def get_network_connector(a2a_network_path: Optional[str] = None,
                         registry_url: Optional[str] = None,
                         trust_service_url: Optional[str] = None) -> NetworkConnector:
    """Get global network connector instance"""
    global _network_connector

    if _network_connector is None:
        _network_connector = NetworkConnector(a2a_network_path, registry_url, trust_service_url)

    return _network_connector
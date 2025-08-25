"""
Standard Trust Relationships Mixin for A2A Agents
Ensures all agents have proper trust relationships with essential management services
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class StandardTrustRelationshipsMixin:
    """
    Mixin that provides standard trust relationships with essential A2A services.
    All agents MUST use this mixin to ensure proper trust compliance.

    Essential Services:
    - data_manager: For data persistence and retrieval
    - agent_manager: For agent registration and lifecycle management
    - catalog_manager: For service discovery and capability registration
    """

    def __init__(self):
        """Initialize standard trust relationships"""
        # Essential service agent IDs
        self.data_manager_agent = "data_manager_agent"
        self.agent_manager_agent = "agent_manager_agent"
        self.catalog_manager_agent = "catalog_manager_agent"

        # Track trust relationship status
        self._trust_relationships_established = False
        self._registered_with_manager = False
        self._registered_in_catalog = False

        logger.info(f"StandardTrustRelationshipsMixin initialized for {getattr(self, 'agent_id', 'unknown')}")

    async def establish_standard_trust_relationships(self) -> bool:
        """
        Establish all standard trust relationships required for A2A compliance.
        This method MUST be called during agent initialization.
        """
        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            logger.info(f"Establishing standard trust relationships for {agent_id}")

            # 1. Register with Agent Manager
            await self._register_with_agent_manager()

            # 2. Register in Catalog Manager
            await self._register_in_catalog()

            # 3. Establish data persistence connection
            await self._establish_data_manager_connection()

            self._trust_relationships_established = True
            logger.info(f"✅ All standard trust relationships established for {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to establish standard trust relationships: {e}")
            return False

    async def _register_with_agent_manager(self) -> bool:
        """Register this agent with the agent manager"""
        try:
            # Prepare agent information
            agent_info = {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "name": getattr(self, 'name', 'Unknown Agent'),
                "description": getattr(self, 'description', ''),
                "version": getattr(self, 'version', '1.0.0'),
                "capabilities": list(getattr(self, 'skills', {}).keys()),
                "status": "active",
                "registered_at": datetime.utcnow().isoformat()
            }

            # Register with agent manager via A2A protocol
            result = await self.call_agent_skill_a2a(
                target_agent=self.agent_manager_agent,
                skill_name="register_agent",
                input_data=agent_info,
                context_id=f"registration_{agent_info['agent_id']}",
                encrypt_data=False
            )

            if result.get("success"):
                self._registered_with_manager = True
                logger.info(f"✅ Registered with agent manager: {agent_info['agent_id']}")

                # Send periodic heartbeats
                asyncio.create_task(self._send_heartbeats())
                return True
            else:
                logger.error(f"Failed to register with agent manager: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error registering with agent manager: {e}")
            return False

    async def _register_in_catalog(self) -> bool:
        """Register agent services in the catalog manager"""
        try:
            # Prepare service catalog entry
            catalog_entry = {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "services": [],
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "tags": getattr(self, 'tags', []),
                    "category": getattr(self, 'category', 'general')
                }
            }

            # Add all skills as services
            for skill_name, skill_func in getattr(self, 'skills', {}).items():
                service_info = {
                    "name": skill_name,
                    "description": getattr(skill_func, 'description', ''),
                    "input_schema": getattr(skill_func, 'input_schema', {}),
                    "output_schema": getattr(skill_func, 'output_schema', {}),
                    "version": "1.0.0"
                }
                catalog_entry["services"].append(service_info)

            # Register in catalog via A2A protocol
            result = await self.call_agent_skill_a2a(
                target_agent=self.catalog_manager_agent,
                skill_name="register_services",
                input_data=catalog_entry,
                context_id=f"catalog_registration_{catalog_entry['agent_id']}",
                encrypt_data=False
            )

            if result.get("success"):
                self._registered_in_catalog = True
                logger.info(f"✅ Registered in catalog manager: {catalog_entry['agent_id']}")
                return True
            else:
                logger.error(f"Failed to register in catalog: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error registering in catalog: {e}")
            return False

    async def _establish_data_manager_connection(self) -> bool:
        """Establish connection with data manager for persistence"""
        try:
            # Initialize data storage namespace
            storage_init = {
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "namespace": f"agent_{getattr(self, 'agent_id', 'unknown')}_data",
                "storage_type": "persistent",
                "schema": {
                    "results": {"type": "array"},
                    "state": {"type": "object"},
                    "metrics": {"type": "object"}
                }
            }

            # Initialize storage via A2A protocol
            result = await self.call_agent_skill_a2a(
                target_agent=self.data_manager_agent,
                skill_name="initialize_storage",
                input_data=storage_init,
                context_id=f"storage_init_{storage_init['agent_id']}",
                encrypt_data=False
            )

            if result.get("success"):
                logger.info(f"✅ Data manager connection established: {storage_init['agent_id']}")
                return True
            else:
                logger.error(f"Failed to establish data manager connection: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error establishing data manager connection: {e}")
            return False

    async def _send_heartbeats(self):
        """Send periodic heartbeats to agent manager"""
        while True:
            try:
                await asyncio.sleep(60)  # Send heartbeat every minute

                heartbeat_data = {
                    "agent_id": getattr(self, 'agent_id', 'unknown'),
                    "status": "active",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "memory_usage": "N/A",  # Would be actual metrics in production
                        "cpu_usage": "N/A",
                        "active_tasks": 0
                    }
                }

                # Send heartbeat via A2A protocol
                await self.call_agent_skill_a2a(
                    target_agent=self.agent_manager_agent,
                    skill_name="heartbeat",
                    input_data=heartbeat_data,
                    context_id=f"heartbeat_{heartbeat_data['agent_id']}",
                    encrypt_data=False
                )

            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    # Standard trust relationship methods that all agents should use

    async def store_agent_data(self, data_type: str, data: Dict[str, Any]) -> bool:
        """
        Store data via A2A protocol to data_manager.
        All agents MUST use this method for data persistence.
        """
        try:
            result = await self.call_agent_skill_a2a(
                target_agent=self.data_manager_agent,
                skill_name="store_data",
                input_data={
                    "data_type": data_type,
                    "data": data,
                    "source_agent": getattr(self, 'agent_id', 'unknown'),
                    "timestamp": datetime.utcnow().isoformat()
                },
                encrypt_data=True
            )
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False

    async def retrieve_agent_data(self, data_type: str, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Retrieve data via A2A protocol from data_manager.
        All agents MUST use this method for data retrieval.
        """
        try:
            result = await self.call_agent_skill_a2a(
                target_agent=self.data_manager_agent,
                skill_name="retrieve_data",
                input_data={
                    "data_type": data_type,
                    "query": query or {},
                    "source_agent": getattr(self, 'agent_id', 'unknown')
                },
                encrypt_data=True
            )
            if result.get("success"):
                return result.get("data")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None

    async def discover_agents(self, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Discover other agents via catalog_manager.
        All agents MUST use this method for service discovery.
        """
        try:
            result = await self.call_agent_skill_a2a(
                target_agent=self.catalog_manager_agent,
                skill_name="discover_agents",
                input_data={
                    "capability": capability,
                    "requester": getattr(self, 'agent_id', 'unknown')
                },
                encrypt_data=False
            )
            if result.get("success"):
                return result.get("agents", [])
            return []
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return []

    async def update_agent_status(self, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update agent status with agent_manager.
        All agents MUST use this method for status updates.
        """
        try:
            result = await self.call_agent_skill_a2a(
                target_agent=self.agent_manager_agent,
                skill_name="update_status",
                input_data={
                    "agent_id": getattr(self, 'agent_id', 'unknown'),
                    "status": status,
                    "details": details or {},
                    "timestamp": datetime.utcnow().isoformat()
                },
                encrypt_data=False
            )
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False

    def verify_trust_compliance(self) -> Dict[str, bool]:
        """
        Verify that all standard trust relationships are properly established.
        Returns compliance status for each essential service.
        """
        return {
            "overall_compliance": self._trust_relationships_established,
            "agent_manager": self._registered_with_manager,
            "catalog_manager": self._registered_in_catalog,
            "data_manager": hasattr(self, 'data_manager_agent'),
            "has_store_method": hasattr(self, 'store_agent_data'),
            "has_retrieve_method": hasattr(self, 'retrieve_agent_data'),
            "has_discover_method": hasattr(self, 'discover_agents'),
            "has_status_method": hasattr(self, 'update_agent_status')
        }

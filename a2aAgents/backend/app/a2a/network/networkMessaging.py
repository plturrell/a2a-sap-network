"""
Network Messaging Service - Agent-to-agent messaging through a2aNetwork
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime

from .networkConnector import get_network_connector

logger = logging.getLogger(__name__)


class NetworkMessagingService:
    """
    Service for agent-to-agent messaging through a2aNetwork
    Provides reliable message delivery with fallback mechanisms
    """

    def __init__(self):
        self.message_handlers = {}
        self.message_queue = {}
        self.network_connector = None
        self.message_id_counter = 0

    async def initialize(self):
        """Initialize the messaging service"""
        try:
            self.network_connector = get_network_connector()
            await self.network_connector.initialize()
            logger.info("NetworkMessagingService initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NetworkMessagingService: {e}")

    async def send_agent_message(self, from_agent_id: str, to_agent_id: str,
                                message_type: str, payload: Dict[str, Any],
                                context_id: str = None) -> Dict[str, Any]:
        """
        Send message from one agent to another through the network

        Args:
            from_agent_id: Sender agent ID
            to_agent_id: Recipient agent ID
            message_type: Type of message (e.g., 'task_request', 'data_query')
            payload: Message payload data
            context_id: Optional context ID for workflow tracking

        Returns:
            Message delivery result
        """
        try:
            if not self.network_connector:
                await self.initialize()

            # Create message structure
            message = {
                "message_id": self._generate_message_id(),
                "from_agent": from_agent_id,
                "to_agent": to_agent_id,
                "message_type": message_type,
                "payload": payload,
                "context_id": context_id or self._generate_context_id(),
                "timestamp": datetime.utcnow().isoformat(),
                "protocol_version": "2.9"
            }

            logger.info(f"Sending A2A message: {from_agent_id} -> {to_agent_id} [{message_type}]")

            # Send through network
            result = await self.network_connector.send_message(
                from_agent_id, to_agent_id, message
            )

            if result.get("success", False):
                logger.info(f"✅ Message delivered successfully: {message['message_id']}")
            else:
                logger.warning(f"⚠️  Message delivery failed: {result}")

            return result

        except Exception as e:
            logger.error(f"Error sending agent message: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def broadcast_message(self, from_agent_id: str, message_type: str,
                              payload: Dict[str, Any], capabilities: List[str] = None) -> List[Dict[str, Any]]:
        """
        Broadcast message to agents with specific capabilities

        Args:
            from_agent_id: Sender agent ID
            message_type: Type of message
            payload: Message payload
            capabilities: Required capabilities for recipients

        Returns:
            List of delivery results
        """
        try:
            if not self.network_connector:
                await self.initialize()

            # Find target agents
            target_agents = await self.network_connector.find_agents(capabilities=capabilities)

            if not target_agents:
                logger.warning(f"No agents found with capabilities: {capabilities}")
                return []

            logger.info(f"Broadcasting to {len(target_agents)} agents with capabilities: {capabilities}")

            # Send to all target agents
            results = []
            for agent in target_agents:
                agent_id = agent.get("agent_id")
                if agent_id and agent_id != from_agent_id:  # Don't send to self
                    result = await self.send_agent_message(
                        from_agent_id, agent_id, message_type, payload
                    )
                    results.append({
                        "target_agent": agent_id,
                        "result": result
                    })

            return results

        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            return []

    async def request_agent_capability(self, from_agent_id: str, capability: str,
                                     request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Request a specific capability from any available agent

        Args:
            from_agent_id: Requesting agent ID
            capability: Required capability
            request_data: Data for the capability request

        Returns:
            Response from capable agent, or None if none available
        """
        try:
            if not self.network_connector:
                await self.initialize()

            # Find agents with the required capability
            capable_agents = await self.network_connector.find_agents(capabilities=[capability])

            if not capable_agents:
                logger.warning(f"No agents found with capability: {capability}")
                return None

            # Try each capable agent until one responds
            for agent in capable_agents:
                agent_id = agent.get("agent_id")
                if agent_id and agent_id != from_agent_id:
                    try:
                        result = await self.send_agent_message(
                            from_agent_id, agent_id, "capability_request", {
                                "requested_capability": capability,
                                "request_data": request_data
                            }
                        )

                        if result.get("success", False):
                            logger.info(f"✅ Capability request fulfilled by {agent_id}")
                            return result

                    except Exception as e:
                        logger.warning(f"Failed to get capability from {agent_id}: {e}")
                        continue

            logger.warning(f"No agents could fulfill capability request: {capability}")
            return None

        except Exception as e:
            logger.error(f"Error requesting capability: {e}")
            return None

    async def register_message_handler(self, agent_id: str, message_type: str,
                                     handler: Callable) -> bool:
        """
        Register handler for incoming messages of specific type

        Args:
            agent_id: Agent ID to register handler for
            message_type: Type of messages to handle
            handler: Async function to handle messages

        Returns:
            True if registered successfully
        """
        try:
            if agent_id not in self.message_handlers:
                self.message_handlers[agent_id] = {}

            self.message_handlers[agent_id][message_type] = handler

            logger.info(f"Message handler registered: {agent_id} -> {message_type}")
            return True

        except Exception as e:
            logger.error(f"Error registering message handler: {e}")
            return False

    async def handle_incoming_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming message from network

        Args:
            message: Incoming message

        Returns:
            Handler response
        """
        try:
            to_agent = message.get("to_agent")
            message_type = message.get("message_type")

            if not to_agent or not message_type:
                return {
                    "success": False,
                    "error": "Invalid message format"
                }

            # Find handler
            handler = self.message_handlers.get(to_agent, {}).get(message_type)

            if not handler:
                logger.warning(f"No handler found for {to_agent}:{message_type}")
                return {
                    "success": False,
                    "error": f"No handler for message type: {message_type}"
                }

            # Execute handler
            logger.info(f"Handling message: {message.get('message_id')} -> {to_agent}:{message_type}")

            result = await handler(message)

            return {
                "success": True,
                "response": result,
                "handled_by": to_agent,
                "message_id": message.get("message_id")
            }

        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_message_status(self, message_id: str) -> Dict[str, Any]:
        """
        Get status of a sent message

        Args:
            message_id: Message identifier

        Returns:
            Message status information
        """
        try:
            # In a full implementation, this would query the network for message status
            # For now, return basic status
            return {
                "message_id": message_id,
                "status": "delivered",  # Mock status
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting message status: {e}")
            return {
                "message_id": message_id,
                "status": "error",
                "error": str(e)
            }

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self.message_id_counter += 1
        return f"msg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.message_id_counter:04d}"

    def _generate_context_id(self) -> str:
        """Generate unique context ID"""
        return f"ctx_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.message_id_counter:04d}"

    async def shutdown(self):
        """Clean shutdown of messaging service"""
        try:
            # Clear message handlers
            self.message_handlers.clear()

            # Clear message queue
            self.message_queue.clear()

            logger.info("NetworkMessagingService shutdown complete")

        except Exception as e:
            logger.error(f"NetworkMessagingService shutdown error: {e}")


# Global messaging service instance
_messaging_service = None

def get_messaging_service() -> NetworkMessagingService:
    """Get global messaging service instance"""
    global _messaging_service

    if _messaging_service is None:
        _messaging_service = NetworkMessagingService()

    return _messaging_service
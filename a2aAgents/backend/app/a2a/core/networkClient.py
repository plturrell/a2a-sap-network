"""
A2A Protocol Compliant Network Communication Layer
All network communication MUST go through A2A blockchain messages

This module enforces A2A protocol compliance by routing all network
requests through the blockchain-based A2A messaging system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """A2A Protocol Types Only"""
    A2A_BLOCKCHAIN = "a2a_blockchain"
    A2A_ENCRYPTED = "a2a_encrypted"
    A2A_CONSENSUS = "a2a_consensus"


@dataclass
class NetworkConfig:
    """A2A Network configuration"""
    blockchain_timeout: int = 60  # seconds for blockchain confirmations
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    require_consensus: bool = False
    consensus_threshold: float = 0.51  # 51% for simple majority
    enable_encryption: bool = True


@dataclass
class NetworkRequest:
    """A2A Network request specification"""
    to_agent: str  # Target agent ID
    message_type: str = "data_request"
    payload: Dict[str, Any] = field(default_factory=dict)
    require_receipt: bool = True
    encrypt: bool = True
    priority: int = 1  # 1-5, higher is more important


class A2ANetworkClient:
    """
    A2A Protocol Compliant Network Client

    This client ensures all network communication goes through
    the A2A blockchain messaging protocol. Direct HTTP/WebSocket
    connections are NOT allowed.
    """

    def __init__(self, agent_id: str, blockchain_client: Any, config: NetworkConfig = None):
        self.agent_id = agent_id
        self.blockchain_client = blockchain_client
        self.config = config or NetworkConfig()

        if not blockchain_client:
            raise ValueError(
                "Blockchain client is required for A2A protocol compliance. "
                "All network communication must go through the A2A blockchain."
            )

    async def send_request(self, request: NetworkRequest) -> Dict[str, Any]:
        """
        Send a request through A2A blockchain messaging

        This replaces traditional HTTP/WebSocket requests with
        blockchain-based A2A protocol messages.
        """
        if not self.blockchain_client:
            raise RuntimeError("Cannot send network request without blockchain client")

        # Create A2A message
        a2a_message = {
            "from": self.agent_id,
            "to": request.to_agent,
            "type": request.message_type,
            "payload": request.payload,
            "timestamp": datetime.utcnow().isoformat(),
            "require_receipt": request.require_receipt,
            "encrypted": request.encrypt and self.config.enable_encryption,
            "priority": request.priority
        }

        # Send through blockchain
        try:
            tx_hash = await self.blockchain_client.send_a2a_message(
                to_address=request.to_agent,
                message=a2a_message,
                encrypt=request.encrypt
            )

            if request.require_receipt:
                # Wait for blockchain confirmation
                receipt = await self.blockchain_client.wait_for_receipt(
                    tx_hash,
                    timeout=self.config.blockchain_timeout
                )

                return {
                    "success": True,
                    "tx_hash": tx_hash,
                    "receipt": receipt,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": True,
                    "tx_hash": tx_hash,
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            raise RuntimeError(f"A2A protocol error: {e}")

    async def send_a2a_message(self, to_agent: str, message: Dict[str, Any], message_type: str = "GENERAL") -> Dict[str, Any]:
        """
        Send an A2A message - alias for send_request for backward compatibility
        """
        request = NetworkRequest(
            to_agent=to_agent,
            message_type=message_type,
            payload=message
        )
        return await self.send_request(request)

    async def request_with_consensus(
        self,
        request: NetworkRequest,
        validator_agents: List[str]
    ) -> Dict[str, Any]:
        """
        Send request requiring consensus from multiple agents

        This ensures distributed validation through the A2A network
        """
        if not validator_agents:
            raise ValueError("Validator agents required for consensus request")

        # Request consensus through blockchain
        consensus_request = {
            "initiator": self.agent_id,
            "request": request,
            "validators": validator_agents,
            "threshold": self.config.consensus_threshold,
            "timeout": self.config.blockchain_timeout
        }

        try:
            result = await self.blockchain_client.request_consensus(consensus_request)

            return {
                "success": result.get("consensus_reached", False),
                "votes": result.get("votes", {}),
                "threshold": self.config.consensus_threshold,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Consensus request failed: {e}")
            raise RuntimeError(f"A2A consensus error: {e}")

    async def close(self):
        """Close network client connections"""
        # No persistent connections in A2A protocol
        # All communication is message-based through blockchain
        pass


# Legacy compatibility wrapper - logs warnings
class LegacyHTTPClient:
    """
    Legacy HTTP client wrapper that enforces A2A protocol

    This class exists only to catch legacy HTTP usage and
    redirect it through proper A2A channels.
    """

    def __init__(self, *args, **kwargs):
        logger.error(
            "PROTOCOL VIOLATION: Attempted to create HTTP client. "
            "All network communication must use A2A blockchain protocol. "
            "Use A2ANetworkClient instead."
        )
        raise RuntimeError(
            "Direct HTTP communication is not allowed in A2A protocol compliant systems. "
            "All agent communication must go through blockchain-based A2A messages."
        )

    async def get(self, *args, **kwargs):
        raise RuntimeError("HTTP GET not allowed. Use A2A protocol.")

    async def post(self, *args, **kwargs):
        raise RuntimeError("HTTP POST not allowed. Use A2A protocol.")

    async def put(self, *args, **kwargs):
        raise RuntimeError("HTTP PUT not allowed. Use A2A protocol.")

    async def delete(self, *args, **kwargs):
        raise RuntimeError("HTTP DELETE not allowed. Use A2A protocol.")


# Replace any imports of httpx/aiohttp/requests with protocol violation error
def __getattr__(name):
    if name in ['httpx', 'aiohttp', 'requests', 'HTTPClient', 'AsyncHTTPClient']:
        raise RuntimeError(
            f"PROTOCOL VIOLATION: Attempted to use {name}. "
            f"Direct HTTP libraries are not allowed in A2A protocol compliant systems. "
            f"All communication must go through blockchain-based A2A messages."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
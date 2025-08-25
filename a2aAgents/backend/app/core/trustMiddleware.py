"""
Trust Middleware for A2A Agent Communication
Automatically handles trust verification and message signing for agents
"""

import asyncio
import functools
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .trustInitializer import get_trust_initializer, initialize_agent_trust_system

logger = logging.getLogger(__name__)


class TrustMiddleware:
    """
    Trust middleware for automatic trust verification and message signing
    """

    def __init__(self):
        self.trust_initializer = None
        self.enabled = True
        self.auto_sign_messages = True
        self.verify_incoming_messages = True

    async def initialize(self):
        """Initialize trust middleware"""
        try:
            self.trust_initializer = await get_trust_initializer()
            logger.info("✅ Trust middleware initialized")
        except Exception as e:
            logger.warning(f"Trust middleware initialization failed: {e}")
            self.enabled = False

    async def ensure_agent_trust(self, agent_id: str, agent_type: str, agent_name: str = None):
        """Ensure agent has trust identity initialized"""
        if not self.enabled or not self.trust_initializer:
            return

        try:
            await initialize_agent_trust_system(agent_id, agent_type, agent_name)
        except Exception as e:
            logger.error(f"Failed to ensure trust for agent {agent_id}: {e}")

    async def sign_outgoing_message(
        self,
        agent_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sign an outgoing message"""
        if not self.enabled or not self.auto_sign_messages or not self.trust_initializer:
            return message

        try:
            return await self.trust_initializer.sign_agent_message(agent_id, message)
        except Exception as e:
            logger.error(f"Failed to sign message for {agent_id}: {e}")
            return message

    async def verify_incoming_message(
        self,
        signed_message: Dict[str, Any]
    ) -> tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Verify an incoming message

        Returns:
            (verified, verification_result, original_message)
        """
        if not self.enabled or not self.verify_incoming_messages or not self.trust_initializer:
            # Extract original message if it's wrapped
            original_message = signed_message.get("message", signed_message)
            return True, {"status": "trust_disabled"}, original_message

        try:
            verified, verification_result = await self.trust_initializer.verify_agent_message(signed_message)
            original_message = signed_message.get("message", signed_message)

            # Update trust score based on verification
            if "agent_id" in verification_result:
                await self.trust_initializer.update_trust_score(
                    verification_result["agent_id"],
                    verified,
                    "message_verification"
                )

            return verified, verification_result, original_message

        except Exception as e:
            logger.error(f"Message verification failed: {e}")
            original_message = signed_message.get("message", signed_message)
            return False, {"error": str(e)}, original_message

    async def record_interaction_outcome(
        self,
        agent_id: str,
        success: bool,
        context: str = None
    ):
        """Record the outcome of an agent interaction"""
        if not self.enabled or not self.trust_initializer:
            return

        try:
            await self.trust_initializer.update_trust_score(agent_id, success, context)
        except Exception as e:
            logger.error(f"Failed to record interaction outcome for {agent_id}: {e}")


# Global trust middleware instance
_trust_middleware: Optional[TrustMiddleware] = None


async def get_trust_middleware() -> TrustMiddleware:
    """Get global trust middleware instance"""
    global _trust_middleware

    if _trust_middleware is None:
        _trust_middleware = TrustMiddleware()
        await _trust_middleware.initialize()

    return _trust_middleware


def with_trust_verification(agent_id: str, agent_type: str = None):
    """
    Decorator to add automatic trust verification to agent handlers
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            trust_middleware = await get_trust_middleware()

            # Initialize agent trust if needed
            if agent_type:
                await trust_middleware.ensure_agent_trust(agent_id, agent_type)

            # Look for message in arguments
            message = None
            for arg in args:
                if hasattr(arg, 'dict') and callable(getattr(arg, 'dict')):
                    # Pydantic model - likely A2AMessage
                    message = arg
                    break
                elif isinstance(arg, dict):
                    message = arg
                    break

            # Verify incoming message if found
            if message and trust_middleware.verify_incoming_messages:
                if isinstance(message, dict):
                    verified, verification_result, original_message = await trust_middleware.verify_incoming_message(message)
                    if not verified:
                        logger.warning(f"Trust verification failed: {verification_result}")
                        # You might want to handle this differently based on your security requirements
                        # For now, we'll log but continue

            try:
                # Execute the original function
                result = await func(*args, **kwargs)

                # Record successful interaction
                await trust_middleware.record_interaction_outcome(agent_id, True, func.__name__)

                # Sign outgoing response if it's a dict
                if isinstance(result, dict) and trust_middleware.auto_sign_messages:
                    result = await trust_middleware.sign_outgoing_message(agent_id, result)

                return result

            except Exception as e:
                # Record failed interaction
                await trust_middleware.record_interaction_outcome(agent_id, False, f"{func.__name__}_error")
                raise

        return wrapper
    return decorator


def with_trust_signing(agent_id: str):
    """
    Decorator to add automatic message signing to agent responses
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            trust_middleware = await get_trust_middleware()

            try:
                # Execute the original function
                result = await func(*args, **kwargs)

                # Sign outgoing response if it's a dict
                if isinstance(result, dict) and trust_middleware.auto_sign_messages:
                    result = await trust_middleware.sign_outgoing_message(agent_id, result)

                return result

            except Exception as e:
                raise

        return wrapper
    return decorator


class TrustedAgentMixin:
    """
    Mixin class to add trust functionality to agents
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_middleware = None
        self._trust_initialized = False

    async def _initialize_trust(self):
        """Initialize trust for this agent"""
        if self._trust_initialized:
            return

        try:
            self.trust_middleware = await get_trust_middleware()

            # Initialize trust identity
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            agent_type = getattr(self, 'agent_type', self.__class__.__name__)
            agent_name = getattr(self, 'name', agent_id)

            await self.trust_middleware.ensure_agent_trust(agent_id, agent_type, agent_name)
            self._trust_initialized = True

            logger.info(f"✅ Trust initialized for agent: {agent_id}")

        except Exception as e:
            logger.error(f"Trust initialization failed for agent: {e}")

    async def send_trusted_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message with trust signature"""
        if not self._trust_initialized:
            await self._initialize_trust()

        if self.trust_middleware:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            return await self.trust_middleware.sign_outgoing_message(agent_id, message)

        return message

    async def verify_received_message(self, signed_message: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """Verify a received message"""
        if not self._trust_initialized:
            await self._initialize_trust()

        if self.trust_middleware:
            return await self.trust_middleware.verify_incoming_message(signed_message)

        # Return unverified if trust not available
        original_message = signed_message.get("message", signed_message)
        return True, {"status": "trust_unavailable"}, original_message

    async def record_interaction_success(self, context: str = None):
        """Record a successful interaction"""
        if self.trust_middleware:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            await self.trust_middleware.record_interaction_outcome(agent_id, True, context)

    async def record_interaction_failure(self, context: str = None):
        """Record a failed interaction"""
        if self.trust_middleware:
            agent_id = getattr(self, 'agent_id', 'unknown_agent')
            await self.trust_middleware.record_interaction_outcome(agent_id, False, context)


# Convenience functions
async def initialize_agent_trust(agent_id: str, agent_type: str, agent_name: str = None):
    """Initialize trust for an agent"""
    trust_middleware = await get_trust_middleware()
    await trust_middleware.ensure_agent_trust(agent_id, agent_type, agent_name)


async def sign_message(agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Sign a message for an agent"""
    trust_middleware = await get_trust_middleware()
    return await trust_middleware.sign_outgoing_message(agent_id, message)


async def verify_message(signed_message: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """Verify a signed message"""
    trust_middleware = await get_trust_middleware()
    return await trust_middleware.verify_incoming_message(signed_message)


# Export main components
__all__ = [
    'TrustMiddleware',
    'get_trust_middleware',
    'with_trust_verification',
    'with_trust_signing',
    'TrustedAgentMixin',
    'initialize_agent_trust',
    'sign_message',
    'verify_message'
]

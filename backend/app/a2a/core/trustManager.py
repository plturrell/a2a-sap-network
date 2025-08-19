"""
Centralized Trust System Manager for A2A Agents

This module provides a clean interface to the trust system with proper
environment-based configuration and error handling.
"""

import os
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class TrustSystemError(Exception):
    """Raised when trust system operations fail"""


class TrustManager:
    """Manages trust system configuration and operations"""

    def __init__(self):
        self.enabled = os.getenv("ENABLE_TRUST", "true").lower() == "true"
        self.required = os.getenv("REQUIRE_TRUST", "false").lower() == "true"
        self.production = os.getenv("ENVIRONMENT", "development").lower() == "production"

        # Import trust functions on initialization
        self._trust_functions = self._import_trust_functions()

        logger.info(
            f"Trust system initialized - enabled: {self.enabled}, required: {self.required}, production: {self.production}"
        )

    def _import_trust_functions(self) -> Dict[str, Callable]:
        """Import trust functions with proper fallback handling"""
        trust_functions = {}

        try:
            # Try relative import first
            from ....a2aNetwork.trustSystem.smartContractTrust import (
                sign_a2a_message,
                initialize_agent_trust,
                verify_a2a_message,
            )

            trust_functions.update(
                {
                    "sign_a2a_message": sign_a2a_message,
                    "initialize_agent_trust": initialize_agent_trust,
                    "verify_a2a_message": verify_a2a_message,
                }
            )
            logger.info("Successfully imported trust functions from a2aNetwork")

        except ImportError:
            try:
                # Try environment-based path
                import sys

                a2a_network_path = os.getenv(
                    "A2A_NETWORK_PATH",
                    os.path.join(os.path.dirname(__file__), "../../../a2aNetwork"),
                )
                sys.path.insert(0, a2a_network_path)
                from trustSystem.smartContractTrust import (
                    sign_a2a_message,
                    initialize_agent_trust,
                    verify_a2a_message,
                )

                trust_functions.update(
                    {
                        "sign_a2a_message": sign_a2a_message,
                        "initialize_agent_trust": initialize_agent_trust,
                        "verify_a2a_message": verify_a2a_message,
                    }
                )
                logger.info("Successfully imported trust functions via environment path")

            except ImportError as e:
                if self.required or self.production:
                    raise TrustSystemError(f"Trust system is required but unavailable: {e}")

                logger.warning(f"Trust system unavailable: {e} - Using development mode")
                # Only provide no-op functions in development when not required
                trust_functions.update(
                    {
                        "sign_a2a_message": self._dev_sign_message,
                        "initialize_agent_trust": self._dev_initialize_trust,
                        "verify_a2a_message": self._dev_verify_message,
                    }
                )

        return trust_functions

    def _ensure_serializable(self, message: Any) -> Any:
        """Ensure message is serializable by converting unhashable types"""
        import json

        try:
            # Test if message is already serializable
            json.dumps(message)
            return message
        except (TypeError, ValueError):
            # Convert to serializable format
            if isinstance(message, dict):
                # Create a deep copy and convert unhashable values to strings
                serializable = {}
                for key, value in message.items():
                    try:
                        json.dumps(value)  # Test if value is serializable
                        serializable[str(key)] = value
                    except (TypeError, ValueError):
                        # Convert unhashable values to string representation
                        serializable[str(key)] = str(value)
                return serializable
            else:
                return str(message)

    def _dev_sign_message(self, message: Any, agent_id: str) -> Dict[str, Any]:
        """Development-only message signing"""
        if self.production:
            raise TrustSystemError("Development trust functions cannot be used in production")
        return {
            "signature": f"dev_signature_{agent_id}",
            "timestamp": "dev_timestamp",
            "agent_id": agent_id,
            "development_mode": True,
        }

    def _dev_initialize_trust(self, agent_id: str, *args, **kwargs) -> Dict[str, Any]:
        """Development-only trust initialization"""
        if self.production:
            raise TrustSystemError("Development trust functions cannot be used in production")
        return {
            "agent_id": agent_id,
            "initialized": True,
            "development_mode": True,
            "trust_address": f"dev_address_{agent_id}",
        }

    def _dev_verify_message(self, message: Any, agent_id: str) -> Dict[str, Any]:
        """Development-only message verification"""
        if self.production:
            raise TrustSystemError("Development trust functions cannot be used in production")
        return {"valid": True, "signer_id": "dev_signer", "development_mode": True}

    def sign_message(self, message: Any, agent_id: str) -> Dict[str, Any]:
        """Sign an A2A message"""
        if not self.enabled:
            return {"signature": "disabled", "agent_id": agent_id}

        try:
            # Fix for unhashable dict error - ensure message is serializable
            serializable_message = self._ensure_serializable(message)
            return self._trust_functions["sign_a2a_message"](serializable_message, agent_id)
        except TypeError as e:
            if "unhashable type" in str(e):
                logger.error(f"Message contains unhashable types: {e}")
                # Try alternative serialization approach
                try:
                    import json

                    serialized = json.dumps(message, default=str, sort_keys=True)
                    message_dict = json.loads(serialized)
                    return self._trust_functions["sign_a2a_message"](message_dict, agent_id)
                except Exception as inner_e:
                    logger.error(f"Failed to serialize message: {inner_e}")
                    if self.required:
                        raise TrustSystemError(
                            f"Message signing failed due to serialization: {inner_e}"
                        )
                    return {
                        "signature": "serialization_failed",
                        "agent_id": agent_id,
                        "error": str(inner_e),
                    }
            else:
                raise  # Re-raise non-serialization TypeErrors
        except Exception as e:
            if self.required:
                raise TrustSystemError(f"Message signing failed and trust is required: {e}")
            logger.warning(f"Message signing failed, continuing without trust: {e}")
            return {"signature": "failed", "agent_id": agent_id, "error": str(e)}

    def initialize_trust(self, agent_id: str, *args, **kwargs) -> Dict[str, Any]:
        """Initialize agent trust system"""
        if not self.enabled:
            return {"initialized": False, "reason": "trust_disabled"}

        try:
            return self._trust_functions["initialize_agent_trust"](agent_id, *args, **kwargs)
        except Exception as e:
            if self.required:
                raise TrustSystemError(f"Trust initialization failed and trust is required: {e}")
            logger.warning(f"Trust initialization failed, continuing without trust: {e}")
            return {"initialized": False, "error": str(e)}

    def verify_message(self, message: Any, agent_id: str) -> Dict[str, Any]:
        """Verify an A2A message"""
        if not self.enabled:
            return {"valid": True, "reason": "trust_disabled"}

        try:
            return self._trust_functions["verify_a2a_message"](message, agent_id)
        except Exception as e:
            if self.required:
                raise TrustSystemError(f"Message verification failed and trust is required: {e}")
            logger.warning(f"Message verification failed, continuing without trust: {e}")
            return {"valid": False, "error": str(e)}

    @property
    def is_available(self) -> bool:
        """Check if trust system is available and functional"""
        return self.enabled and "sign_a2a_message" in self._trust_functions


# Global trust manager instance
trust_manager = TrustManager()


# Convenient exports for backward compatibility
def sign_a2a_message(message: Any, agent_id: str) -> Dict[str, Any]:
    """Sign an A2A message using the global trust manager"""
    return trust_manager.sign_message(message, agent_id)


def initialize_agent_trust(agent_id: str, *args, **kwargs) -> Dict[str, Any]:
    """Initialize agent trust using the global trust manager"""
    return trust_manager.initialize_trust(agent_id, *args, **kwargs)


def verify_a2a_message(message: Any, agent_id: str) -> Dict[str, Any]:
    """Verify an A2A message using the global trust manager"""
    return trust_manager.verify_message(message, agent_id)

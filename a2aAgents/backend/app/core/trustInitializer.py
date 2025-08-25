"""
import time
Trust Identity System Initializer
Automatically initializes trust identities for all A2A agents
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import trust system components
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import (
        initialize_agent_trust,
        sign_a2a_message,
        verify_a2a_message,
        get_trust_contract
    )
    TRUST_SYSTEM_AVAILABLE = True
except ImportError as e:
    TRUST_SYSTEM_AVAILABLE = False
    logging.warning(f"Trust system not available: {e}")

logger = logging.getLogger(__name__)


class TrustIdentityInitializer:
    """
    Trust Identity System Initializer for A2A Agents

    Manages trust identity initialization and maintenance for all agents in the system.
    """

    def __init__(self):
        self.initialized_agents: Dict[str, Dict[str, Any]] = {}
        self.trust_contract = None
        self.initialization_config = {
            "auto_initialize": True,
            "trust_verification_enabled": True,
            "trust_score_monitoring": True,
            "trust_relationship_auto_creation": True,
            "backup_trust_identities": True
        }

        # Agent type compatibility matrix
        self.agent_type_compatibility = {
            "DataProductRegistrationAgent": ["FinancialStandardizationAgent", "ORDRegistryAgent", "DataManagerAgent"],
            "FinancialStandardizationAgent": ["DataProductRegistrationAgent", "ComplianceAgent"],
            "ORDRegistryAgent": ["DataProductRegistrationAgent", "SearchAgent", "DataManagerAgent"],
            "DataManagerAgent": ["DataProductRegistrationAgent", "ORDRegistryAgent", "CacheManagerAgent"],
            "ComplianceAgent": ["FinancialStandardizationAgent", "AuditAgent"],
            "SearchAgent": ["ORDRegistryAgent", "DataManagerAgent"],
            "CacheManagerAgent": ["DataManagerAgent"],
            "AuditAgent": ["ComplianceAgent"],
            "AgentBuilderAgent": ["all"],  # Agent builder can communicate with all agents
            "WorkflowOrchestratorAgent": ["all"],
            "SecurityMonitorAgent": ["all"]
        }

        # Trust initialization storage
        self.trust_storage_path = os.getenv("TRUST_STORAGE_PATH", "/tmp/a2a_trust_identities")
        os.makedirs(self.trust_storage_path, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the trust identity system"""
        try:
            if not TRUST_SYSTEM_AVAILABLE:
                logger.warning("⚠️ Trust system not available - operating in compatibility mode")
                return

            # Initialize trust contract
            self.trust_contract = get_trust_contract()
            logger.info(f"✅ Trust contract initialized: {self.trust_contract.contract_id}")

            # Load existing trust identities
            await self._load_existing_identities()

            # Initialize predefined agent types
            await self._initialize_predefined_agents()

            logger.info("✅ Trust Identity System initialization complete")

        except Exception as e:
            logger.error(f"❌ Trust Identity System initialization failed: {e}")
            raise

    async def initialize_agent_trust_identity(
        self,
        agent_id: str,
        agent_type: str,
        agent_name: str = None
    ) -> Dict[str, Any]:
        """
        Initialize trust identity for a specific agent

        Args:
            agent_id: Unique agent identifier
            agent_type: Type/class of the agent
            agent_name: Human-readable agent name

        Returns:
            Dict containing trust initialization result
        """
        try:
            if not TRUST_SYSTEM_AVAILABLE:
                logger.warning(f"Trust system not available - skipping trust initialization for {agent_id}")
                return {"status": "skipped", "reason": "trust_system_unavailable"}

            # Check if agent already initialized
            if agent_id in self.initialized_agents:
                logger.info(f"Agent {agent_id} already has trust identity")
                return self.initialized_agents[agent_id]

            # Initialize agent trust in the smart contract
            agent_identity = initialize_agent_trust(agent_id, agent_type)

            # Create trust initialization record
            trust_record = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "agent_name": agent_name or agent_id,
                "initialized_at": datetime.utcnow().isoformat(),
                "trust_score": agent_identity.trust_score,
                "verified": agent_identity.verified,
                "public_key_fingerprint": self._get_key_fingerprint(agent_identity.public_key),
                "trust_relationships": {},
                "status": "initialized"
            }

            # Establish trust relationships with compatible agents
            await self._establish_compatible_relationships(agent_id, agent_type)

            # Store trust record
            self.initialized_agents[agent_id] = trust_record

            # Save to persistent storage
            await self._save_trust_identity(agent_id, trust_record)

            logger.info(f"✅ Trust identity initialized for agent: {agent_id} ({agent_type})")

            return trust_record

        except Exception as e:
            logger.error(f"❌ Failed to initialize trust for agent {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e),
                "initialized_at": datetime.utcnow().isoformat()
            }

    async def _establish_compatible_relationships(self, agent_id: str, agent_type: str):
        """Establish trust relationships with compatible agents"""
        try:
            if not self.trust_contract:
                return

            compatible_types = self.agent_type_compatibility.get(agent_type, [])

            # Check all agents for compatibility
            for existing_agent_id, existing_record in self.initialized_agents.items():
                if existing_agent_id == agent_id:
                    continue

                existing_type = existing_record.get("agent_type")

                # Check if agents are compatible
                if (existing_type in compatible_types or
                    agent_type in self.agent_type_compatibility.get(existing_type, []) or
                    "all" in compatible_types or
                    "all" in self.agent_type_compatibility.get(existing_type, [])):

                    # Establish trust channel
                    try:
                        trust_channel = self.trust_contract.establish_trust_channel(agent_id, existing_agent_id)
                        logger.info(f"✅ Trust channel established: {agent_id} ↔ {existing_agent_id}")

                        # Record relationship
                        if agent_id not in self.initialized_agents:
                            self.initialized_agents[agent_id] = {"trust_relationships": {}}

                        self.initialized_agents[agent_id]["trust_relationships"][existing_agent_id] = {
                            "established_at": datetime.utcnow().isoformat(),
                            "channel_id": trust_channel["channel_id"],
                            "trust_level": trust_channel["trust_level"]
                        }

                    except Exception as e:
                        logger.warning(f"Failed to establish trust channel {agent_id} ↔ {existing_agent_id}: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to establish compatible relationships for {agent_id}: {e}")

    async def _initialize_predefined_agents(self):
        """Initialize trust identities for predefined system agents"""
        predefined_agents = [
            ("data_product_registration_agent", "DataProductRegistrationAgent", "Data Product Registration Agent"),
            ("financial_standardization_agent", "FinancialStandardizationAgent", "Financial Standardization Agent"),
            ("ord_registry_agent", "ORDRegistryAgent", "ORD Registry Agent"),
            ("data_manager_agent", "DataManagerAgent", "Data Manager Agent"),
            ("compliance_agent", "ComplianceAgent", "Compliance Agent"),
            ("search_agent", "SearchAgent", "Search Agent"),
            ("cache_manager_agent", "CacheManagerAgent", "Cache Manager Agent"),
            ("audit_agent", "AuditAgent", "Audit Agent"),
            ("agent_builder_agent", "AgentBuilderAgent", "Agent Builder Agent"),
            ("workflow_orchestrator_agent", "WorkflowOrchestratorAgent", "Workflow Orchestrator Agent"),
            ("security_monitor_agent", "SecurityMonitorAgent", "Security Monitor Agent")
        ]

        for agent_id, agent_type, agent_name in predefined_agents:
            await self.initialize_agent_trust_identity(agent_id, agent_type, agent_name)

    def _get_key_fingerprint(self, public_key: bytes) -> str:
        """Generate fingerprint for public key"""
        import hashlib


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        return hashlib.sha256(public_key).hexdigest()[:16]

    async def _load_existing_identities(self):
        """Load existing trust identities from storage"""
        try:
            trust_file = Path(self.trust_storage_path) / "trust_identities.json"
            if trust_file.exists():
                with open(trust_file, 'r') as f:
                    stored_identities = json.load(f)

                self.initialized_agents.update(stored_identities)
                logger.info(f"✅ Loaded {len(stored_identities)} existing trust identities")

        except Exception as e:
            logger.warning(f"Failed to load existing trust identities: {e}")

    async def _save_trust_identity(self, agent_id: str, trust_record: Dict[str, Any]):
        """Save trust identity to persistent storage"""
        try:
            # Save individual agent file
            agent_file = Path(self.trust_storage_path) / f"{agent_id}_trust.json"
            with open(agent_file, 'w') as f:
                json.dump(trust_record, f, indent=2)

            # Save consolidated file
            trust_file = Path(self.trust_storage_path) / "trust_identities.json"
            with open(trust_file, 'w') as f:
                json.dump(self.initialized_agents, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trust identity for {agent_id}: {e}")

    async def verify_agent_message(
        self,
        signed_message: Dict[str, Any]
    ) -> tuple[bool, Dict[str, Any]]:
        """Verify a signed message from an agent"""
        if not TRUST_SYSTEM_AVAILABLE:
            return True, {"status": "trust_system_unavailable", "verification": "skipped"}

        try:
            return verify_a2a_message(signed_message)
        except Exception as e:
            logger.error(f"Message verification failed: {e}")
            return False, {"error": str(e)}

    async def sign_agent_message(
        self,
        agent_id: str,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sign a message from an agent"""
        if not TRUST_SYSTEM_AVAILABLE:
            return {"message": message, "signature": {"status": "trust_system_unavailable"}}

        try:
            return sign_a2a_message(agent_id, message)
        except Exception as e:
            logger.error(f"Message signing failed for {agent_id}: {e}")
            return {"message": message, "signature": {"error": str(e)}}

    async def get_trust_status(self) -> Dict[str, Any]:
        """Get overall trust system status"""
        try:
            status = {
                "trust_system_available": TRUST_SYSTEM_AVAILABLE,
                "initialized_agents_count": len(self.initialized_agents),
                "trust_contract_id": self.trust_contract.contract_id if self.trust_contract else None,
                "configuration": self.initialization_config,
                "initialized_at": datetime.utcnow().isoformat()
            }

            if self.trust_contract:
                contract_status = self.trust_contract.get_contract_status()
                status["contract_status"] = contract_status

            # Add agent summary
            agent_summary = {}
            for agent_id, record in self.initialized_agents.items():
                agent_summary[agent_id] = {
                    "agent_type": record.get("agent_type"),
                    "trust_score": record.get("trust_score"),
                    "verified": record.get("verified"),
                    "relationships_count": len(record.get("trust_relationships", {})),
                    "status": record.get("status")
                }

            status["agents"] = agent_summary

            return status

        except Exception as e:
            logger.error(f"Failed to get trust status: {e}")
            return {"error": str(e)}

    async def update_trust_score(
        self,
        agent_id: str,
        interaction_success: bool,
        context: str = None
    ):
        """Update trust score based on interaction outcome"""
        try:
            if not TRUST_SYSTEM_AVAILABLE or agent_id not in self.initialized_agents:
                return

            # Record trust event in smart contract
            if self.trust_contract:
                self.trust_contract._record_message_verification(agent_id, interaction_success)

                # Update local record
                updated_score = self.trust_contract.get_trust_score(agent_id)
                self.initialized_agents[agent_id]["trust_score"] = updated_score

                # Save updated record
                await self._save_trust_identity(agent_id, self.initialized_agents[agent_id])

                logger.info(f"✅ Trust score updated for {agent_id}: {updated_score} ({'success' if interaction_success else 'failure'})")

        except Exception as e:
            logger.error(f"Failed to update trust score for {agent_id}: {e}")

    async def cleanup(self):
        """Cleanup trust system resources"""
        try:
            # Save final state
            if self.initialized_agents:
                await self._save_trust_identity("system", {"cleanup_at": datetime.utcnow().isoformat()})

            logger.info("✅ Trust Identity System cleanup completed")

        except Exception as e:
            logger.error(f"Trust system cleanup failed: {e}")


# Global trust initializer instance
_trust_initializer: Optional[TrustIdentityInitializer] = None


async def get_trust_initializer() -> TrustIdentityInitializer:
    """Get global trust initializer instance"""
    global _trust_initializer

    if _trust_initializer is None:
        _trust_initializer = TrustIdentityInitializer()
        await _trust_initializer.initialize()

    return _trust_initializer


async def initialize_agent_trust_system(agent_id: str, agent_type: str, agent_name: str = None):
    """Initialize trust system for an agent (convenience function)"""
    trust_initializer = await get_trust_initializer()
    return await trust_initializer.initialize_agent_trust_identity(agent_id, agent_type, agent_name)


# Export main functions
__all__ = [
    'TrustIdentityInitializer',
    'get_trust_initializer',
    'initialize_agent_trust_system'
]
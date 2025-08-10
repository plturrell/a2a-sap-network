"""
Smart Contract-based Trust System for A2A Communication
Implements cryptographic message signing and agent identity verification
"""

import json
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64
import os

logger = logging.getLogger(__name__)


class AgentIdentity:
    """Agent Identity with cryptographic keys"""
    
    def __init__(self, agent_id: str, agent_type: str, public_key: bytes, private_key: bytes = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.public_key = public_key
        self.private_key = private_key
        self.created_at = datetime.utcnow()
        self.trust_score = 1.0
        self.verified = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "public_key": base64.b64encode(self.public_key).decode(),
            "created_at": self.created_at.isoformat(),
            "trust_score": self.trust_score,
            "verified": self.verified
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentIdentity':
        """Create from dictionary"""
        identity = cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            public_key=base64.b64decode(data["public_key"])
        )
        identity.created_at = datetime.fromisoformat(data["created_at"])
        identity.trust_score = data.get("trust_score", 1.0)
        identity.verified = data.get("verified", False)
        return identity


class SmartContractTrust:
    """Smart Contract-based Trust System for A2A Communication"""
    
    def __init__(self, contract_id: str = None):
        self.contract_id = contract_id or f"a2a_trust_{uuid4().hex[:8]}"
        self.agents: Dict[str, AgentIdentity] = {}
        self.trust_relationships: Dict[str, Dict[str, float]] = {}  # agent_id -> {peer_id: trust_score}
        self.message_history: List[Dict[str, Any]] = []
        self.contract_state = {
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "version": "1.0.0",
            "active": True
        }
        
        # Contract rules
        self.trust_rules = {
            "min_trust_score": 0.7,
            "message_expiry_minutes": 30,
            "max_trust_degradation": 0.1,
            "trust_recovery_rate": 0.05,
            "signature_required": True
        }
        
        logger.info(f"✅ Smart Contract Trust System initialized: {self.contract_id}")
    
    def register_agent(
        self, 
        agent_id: str, 
        agent_type: str, 
        public_key: bytes = None,
        private_key: bytes = None
    ) -> AgentIdentity:
        """Register a new agent with cryptographic identity"""
        try:
            # Generate RSA key pair if not provided
            if not public_key or not private_key:
                private_key_obj = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                private_key = private_key_obj.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                )
                public_key = private_key_obj.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            
            # Create agent identity
            identity = AgentIdentity(
                agent_id=agent_id,
                agent_type=agent_type,
                public_key=public_key,
                private_key=private_key
            )
            
            # Register in contract
            self.agents[agent_id] = identity
            self.trust_relationships[agent_id] = {}
            
            # Initialize trust relationships with other agents
            for other_agent_id in self.agents:
                if other_agent_id != agent_id:
                    # Default mutual trust based on agent types
                    if self._are_compatible_agents(agent_type, self.agents[other_agent_id].agent_type):
                        self.trust_relationships[agent_id][other_agent_id] = 0.8
                        self.trust_relationships[other_agent_id][agent_id] = 0.8
                    else:
                        self.trust_relationships[agent_id][other_agent_id] = 0.5
                        self.trust_relationships[other_agent_id][agent_id] = 0.5
            
            self._update_contract_state()
            logger.info(f"✅ Agent registered in trust contract: {agent_id} ({agent_type})")
            
            return identity
            
        except Exception as e:
            logger.error(f"❌ Failed to register agent {agent_id}: {e}")
            raise
    
    def _are_compatible_agents(self, type1: str, type2: str) -> bool:
        """Check if two agent types are compatible for high trust"""
        compatible_pairs = [
            ("DataProductRegistrationAgent", "FinancialStandardizationAgent"),
            ("FinancialStandardizationAgent", "DataProductRegistrationAgent")
        ]
        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs
    
    def sign_message(self, agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Sign an A2A message with agent's private key"""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not registered")
            
            identity = self.agents[agent_id]
            if not identity.private_key:
                raise ValueError(f"Agent {agent_id} has no private key for signing")
            
            # Create message hash
            message_json = json.dumps(message, sort_keys=True)
            message_hash = hashlib.sha256(message_json.encode()).hexdigest()
            
            # Load private key
            private_key_obj = serialization.load_pem_private_key(
                identity.private_key,
                password=None
            )
            
            # Sign the message hash
            signature = private_key_obj.sign(
                message_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Create signed message
            signed_message = {
                "message": message,
                "signature": {
                    "agent_id": agent_id,
                    "message_hash": message_hash,
                    "signature": base64.b64encode(signature).decode(),
                    "timestamp": datetime.utcnow().isoformat(),
                    "contract_id": self.contract_id
                }
            }
            
            logger.info(f"✅ Message signed by agent: {agent_id}")
            return signed_message
            
        except Exception as e:
            logger.error(f"❌ Failed to sign message for agent {agent_id}: {e}")
            raise
    
    def verify_message(self, signed_message: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Verify a signed A2A message"""
        try:
            signature_info = signed_message.get("signature", {})
            sender_id = signature_info.get("agent_id")
            
            if sender_id not in self.agents:
                return False, {"error": f"Unknown agent: {sender_id}"}
            
            identity = self.agents[sender_id]
            
            # Verify message hash
            message = signed_message.get("message", {})
            message_json = json.dumps(message, sort_keys=True)
            expected_hash = hashlib.sha256(message_json.encode()).hexdigest()
            
            if expected_hash != signature_info.get("message_hash"):
                return False, {"error": "Message hash mismatch"}
            
            # Verify signature
            try:
                public_key_obj = serialization.load_pem_public_key(identity.public_key)
                signature_bytes = base64.b64decode(signature_info.get("signature", ""))
                
                public_key_obj.verify(
                    signature_bytes,
                    expected_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                # Check message age
                timestamp_str = signature_info.get("timestamp", "")
                message_time = datetime.fromisoformat(timestamp_str)
                age_minutes = (datetime.utcnow() - message_time).total_seconds() / 60
                
                if age_minutes > self.trust_rules["message_expiry_minutes"]:
                    return False, {"error": f"Message expired ({age_minutes:.1f} minutes old)"}
                
                # Check trust score
                trust_score = identity.trust_score
                if trust_score < self.trust_rules["min_trust_score"]:
                    return False, {"error": f"Agent trust score too low: {trust_score}"}
                
                verification_result = {
                    "verified": True,
                    "agent_id": sender_id,
                    "agent_type": identity.agent_type,
                    "trust_score": trust_score,
                    "message_age_minutes": age_minutes,
                    "contract_id": self.contract_id
                }
                
                # Record successful verification
                self._record_message_verification(sender_id, True)
                logger.info(f"✅ Message verified from agent: {sender_id}")
                
                return True, verification_result
                
            except Exception as verify_error:
                logger.error(f"❌ Signature verification failed: {verify_error}")
                self._record_message_verification(sender_id, False)
                return False, {"error": f"Invalid signature: {verify_error}"}
            
        except Exception as e:
            logger.error(f"❌ Message verification error: {e}")
            return False, {"error": f"Verification failed: {e}"}
    
    def establish_trust_channel(self, agent1_id: str, agent2_id: str) -> Dict[str, Any]:
        """Establish trusted communication channel between two agents"""
        try:
            if agent1_id not in self.agents or agent2_id not in self.agents:
                raise ValueError("Both agents must be registered")
            
            # Generate shared secret for the channel
            shared_secret = os.urandom(32)
            channel_id = f"channel_{uuid4().hex[:8]}"
            
            # Encrypt shared secret with each agent's public key
            agent1_public = serialization.load_pem_public_key(self.agents[agent1_id].public_key)
            agent2_public = serialization.load_pem_public_key(self.agents[agent2_id].public_key)
            
            encrypted_secret_1 = agent1_public.encrypt(
                shared_secret,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            encrypted_secret_2 = agent2_public.encrypt(
                shared_secret,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Create trust channel
            trust_channel = {
                "channel_id": channel_id,
                "agent1_id": agent1_id,
                "agent2_id": agent2_id,
                "established_at": datetime.utcnow().isoformat(),
                "encrypted_secrets": {
                    agent1_id: base64.b64encode(encrypted_secret_1).decode(),
                    agent2_id: base64.b64encode(encrypted_secret_2).decode()
                },
                "trust_level": "high",
                "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
            }
            
            # Update trust scores
            if agent1_id in self.trust_relationships:
                self.trust_relationships[agent1_id][agent2_id] = min(1.0, 
                    self.trust_relationships[agent1_id].get(agent2_id, 0.5) + 0.2)
            if agent2_id in self.trust_relationships:
                self.trust_relationships[agent2_id][agent1_id] = min(1.0,
                    self.trust_relationships[agent2_id].get(agent1_id, 0.5) + 0.2)
            
            self._update_contract_state()
            logger.info(f"✅ Trust channel established: {channel_id} ({agent1_id} ↔ {agent2_id})")
            
            return trust_channel
            
        except Exception as e:
            logger.error(f"❌ Failed to establish trust channel: {e}")
            raise
    
    def _record_message_verification(self, agent_id: str, success: bool):
        """Record message verification outcome and update trust scores"""
        try:
            verification_record = {
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "success": success,
                "contract_id": self.contract_id
            }
            
            self.message_history.append(verification_record)
            
            # Update trust score based on verification outcome
            if agent_id in self.agents:
                identity = self.agents[agent_id]
                if success:
                    # Increase trust score slightly for successful verifications
                    identity.trust_score = min(1.0, identity.trust_score + self.trust_rules["trust_recovery_rate"])
                else:
                    # Decrease trust score for failed verifications
                    identity.trust_score = max(0.0, identity.trust_score - self.trust_rules["max_trust_degradation"])
                    logger.warning(f"⚠️ Trust score decreased for agent {agent_id}: {identity.trust_score}")
            
            # Keep only recent history (last 1000 records)
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
            
        except Exception as e:
            logger.error(f"❌ Failed to record verification: {e}")
    
    def get_trust_score(self, agent_id: str, peer_id: str = None) -> float:
        """Get trust score for an agent or between two agents"""
        try:
            if agent_id not in self.agents:
                return 0.0
            
            if peer_id is None:
                # Return agent's overall trust score
                return self.agents[agent_id].trust_score
            else:
                # Return mutual trust score between agents
                if agent_id in self.trust_relationships and peer_id in self.trust_relationships[agent_id]:
                    return self.trust_relationships[agent_id][peer_id]
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Failed to get trust score: {e}")
            return 0.0
    
    def _update_contract_state(self):
        """Update contract state timestamp"""
        self.contract_state["last_updated"] = datetime.utcnow()
    
    def get_contract_status(self) -> Dict[str, Any]:
        """Get current contract status and statistics"""
        try:
            # Calculate statistics
            total_agents = len(self.agents)
            verified_agents = sum(1 for agent in self.agents.values() if agent.verified)
            avg_trust_score = sum(agent.trust_score for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
            
            recent_verifications = [
                record for record in self.message_history[-100:]
                if (datetime.utcnow() - datetime.fromisoformat(record["timestamp"])).total_seconds() < 3600
            ]
            
            success_rate = sum(1 for record in recent_verifications if record["success"]) / len(recent_verifications) if recent_verifications else 0
            
            return {
                "contract_id": self.contract_id,
                "status": "active" if self.contract_state["active"] else "inactive",
                "version": self.contract_state["version"],
                "created_at": self.contract_state["created_at"].isoformat(),
                "last_updated": self.contract_state["last_updated"].isoformat(),
                "statistics": {
                    "total_agents": total_agents,
                    "verified_agents": verified_agents,
                    "average_trust_score": round(avg_trust_score, 3),
                    "total_relationships": sum(len(relations) for relations in self.trust_relationships.values()),
                    "recent_verifications": len(recent_verifications),
                    "success_rate": round(success_rate, 3)
                },
                "trust_rules": self.trust_rules
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get contract status: {e}")
            return {"error": str(e)}
    
    def export_agent_identity(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Export agent identity for secure storage or transfer"""
        try:
            if agent_id not in self.agents:
                return None
            
            identity = self.agents[agent_id]
            return {
                "agent_identity": identity.to_dict(),
                "trust_relationships": self.trust_relationships.get(agent_id, {}),
                "contract_id": self.contract_id,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export agent identity: {e}")
            return None


# Global trust contract instance
_trust_contract = None

def get_trust_contract() -> SmartContractTrust:
    """Get or create the global trust contract"""
    # Use shared trust contract to ensure all processes share the same instance
    from .sharedTrust import get_shared_trust_contract
    return get_shared_trust_contract()


def initialize_agent_trust(agent_id: str, agent_type: str) -> AgentIdentity:
    """Initialize trust for a new agent"""
    trust_contract = get_trust_contract()
    return trust_contract.register_agent(agent_id, agent_type)


def sign_a2a_message(agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Sign an A2A message for trusted communication"""
    trust_contract = get_trust_contract()
    return trust_contract.sign_message(agent_id, message)


def verify_a2a_message(signed_message: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Verify a signed A2A message"""
    trust_contract = get_trust_contract()
    return trust_contract.verify_message(signed_message)
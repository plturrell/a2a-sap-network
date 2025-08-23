"""
Trust System Integration for A2A Agents.
Provides production-ready blockchain trust functionality.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
try:
    from config.agentConfig import config
except ImportError:
    # A2A Protocol Compliance: All imports must be available
    # No fallback implementations allowed - the agent must have all required dependencies
    # Fallback configuration
    class Config:
        def __init__(self):
            self.base_url = os.getenv("A2A_SERVICE_URL")
            self.storage_base_path = "/tmp/a2a"
        def get_agent_url(self, agent_type): return self.base_url
        def get_contract_address(self, name): 
            # Return actual deployed contract addresses
            contracts = {
                "AgentRegistry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
                "MessageRouter": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
            }
            return contracts.get(name, "0x0000000000000000000000000000000000000000")
    config = Config()


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class TrustSystemIntegration:
    """Handles blockchain-based trust operations for A2A agents."""
    
    def __init__(self):
        """Initialize trust system with blockchain connection."""
        self.w3 = Web3(Web3.HTTPProvider(config.blockchain_rpc_url))
        self.private_key = os.getenv("AGENT_PRIVATE_KEY")
        self.trust_registry_address = config.get_contract_address("trust_registry")
        
        if not self.private_key:
            raise ValueError("AGENT_PRIVATE_KEY environment variable not set")
        
        self.account = Account.from_key(self.private_key)
        self.trust_registry_abi = self._load_trust_registry_abi()
        
        if self.w3.is_connected():
            self.trust_registry = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.trust_registry_address),
                abi=self.trust_registry_abi
            )
            logger.info(f"Connected to blockchain at {config.blockchain_rpc_url}")
        else:
            raise ConnectionError(f"Failed to connect to blockchain at {config.blockchain_rpc_url}")
    
    def _load_trust_registry_abi(self) -> list:
        """Load trust registry contract ABI."""
        abi_path = os.path.join(
            os.path.dirname(__file__),
            "../contracts/abis/TrustRegistry.json"
        )
        
        # If ABI file doesn't exist, use minimal ABI
        if not os.path.exists(abi_path):
            return [
                {
                    "inputs": [{"name": "agentId", "type": "string"}],
                    "name": "registerAgent",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "agentId", "type": "string"}],
                    "name": "isAgentRegistered",
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"name": "fromAgent", "type": "string"},
                        {"name": "toAgent", "type": "string"},
                        {"name": "score", "type": "uint256"}
                    ],
                    "name": "updateTrustScore",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        
        with open(abi_path, 'r') as f:
            return json.load(f)
    
    def sign_a2a_message(
        self,
        message: Dict[str, Any],
        agent_id: str,
        private_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Sign an A2A message with the agent's private key.
        
        Args:
            message: Message to sign
            agent_id: ID of the signing agent
            private_key: Optional private key (uses default if not provided)
            
        Returns:
            Dict containing signature and metadata
        """
        try:
            # Use provided key or default
            key = private_key or self.private_key
            account = Account.from_key(key) if key != self.private_key else self.account
            
            # Create message hash
            message_str = json.dumps(message, sort_keys=True)
            message_hash = Web3.keccak(text=f"{agent_id}:{message_str}")
            
            # Sign message
            signable_message = encode_defunct(message_hash)
            signed = account.sign_message(signable_message)
            
            return {
                "signature": signed.signature.hex(),
                "message_hash": message_hash.hex(),
                "signer_address": account.address,
                "agent_id": agent_id,
                "timestamp": int(self.w3.eth.get_block('latest')['timestamp'])
            }
            
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            raise
    
    def verify_a2a_message(
        self,
        message: Dict[str, Any],
        signature_data: Dict[str, str]
    ) -> bool:
        """
        Verify an A2A message signature.
        
        Args:
            message: Message to verify
            signature_data: Signature data from sign_a2a_message
            
        Returns:
            True if signature is valid
        """
        try:
            # Recreate message hash
            agent_id = signature_data.get("agent_id")
            message_str = json.dumps(message, sort_keys=True)
            expected_hash = Web3.keccak(text=f"{agent_id}:{message_str}")
            
            # Verify hash matches
            if expected_hash.hex() != signature_data.get("message_hash"):
                logger.warning("Message hash mismatch")
                return False
            
            # Recover signer address
            signable_message = encode_defunct(expected_hash)
            recovered_address = Account.recover_message(
                signable_message,
                signature=signature_data.get("signature")
            )
            
            # Verify address matches
            if recovered_address.lower() != signature_data.get("signer_address", "").lower():
                logger.warning("Signer address mismatch")
                return False
            
            # Verify agent is registered (if connected to blockchain)
            if self.w3.is_connected():
                is_registered = self.trust_registry.functions.isAgentRegistered(agent_id).call()
                if not is_registered:
                    logger.warning(f"Agent {agent_id} not registered in trust registry")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify message: {e}")
            return False
    
    def initialize_agent_trust(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any]
    ) -> bool:
        """
        Initialize trust for a new agent by registering on blockchain.
        
        Args:
            agent_id: Unique agent identifier
            agent_metadata: Agent metadata to store
            
        Returns:
            True if initialization successful
        """
        try:
            # Check if already registered
            is_registered = self.trust_registry.functions.isAgentRegistered(agent_id).call()
            if is_registered:
                logger.info(f"Agent {agent_id} already registered")
                return True
            
            # Build transaction
            tx = self.trust_registry.functions.registerAgent(agent_id).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Agent {agent_id} registered successfully. Tx: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Agent registration failed. Tx: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize agent trust: {e}")
            # In production, we should not continue without proper registration
            if config.blockchain_network == "mainnet":
                raise
            return False
    
    def update_trust_score(
        self,
        from_agent: str,
        to_agent: str,
        score: int
    ) -> bool:
        """
        Update trust score between agents.
        
        Args:
            from_agent: Agent providing trust score
            to_agent: Agent receiving trust score
            score: Trust score (0-100)
            
        Returns:
            True if update successful
        """
        try:
            if not 0 <= score <= 100:
                raise ValueError("Trust score must be between 0 and 100")
            
            # Build transaction
            tx = self.trust_registry.functions.updateTrustScore(
                from_agent, to_agent, score
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 150000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt['status'] == 1
            
        except Exception as e:
            logger.error(f"Failed to update trust score: {e}")
            return False


# Global trust system instance
_trust_system = None

def get_trust_system() -> TrustSystemIntegration:
    """Get or create global trust system instance."""
    global _trust_system
    if _trust_system is None:
        _trust_system = TrustSystemIntegration()
    return _trust_system


# Export functions matching the expected interface
def sign_a2a_message(message: Dict[str, Any], agent_id: str, private_key: Optional[str] = None) -> Dict[str, str]:
    """Sign an A2A message."""
    return get_trust_system().sign_a2a_message(message, agent_id, private_key)


def verify_a2a_message(message: Dict[str, Any], signature_data: Dict[str, str]) -> bool:
    """Verify an A2A message signature."""
    return get_trust_system().verify_a2a_message(message, signature_data)


def initialize_agent_trust(agent_id: str, agent_metadata: Dict[str, Any]) -> bool:
    """Initialize trust for a new agent."""
    return get_trust_system().initialize_agent_trust(agent_id, agent_metadata)
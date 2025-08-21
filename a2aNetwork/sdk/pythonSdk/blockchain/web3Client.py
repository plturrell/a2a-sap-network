#!/usr/bin/env python3
"""
Web3 Client for A2A Network Blockchain Integration
Connects finsight_cib agents to a2a_network smart contracts
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.signers.local import LocalAccount
import asyncio
from dataclasses import dataclass
from ..config.contractConfig import get_contract_config, ContractConfigManager
from .nonceManager import NonceManager, TransactionManager

logger = logging.getLogger(__name__)

@dataclass
class AgentIdentity:
    """Agent identity on blockchain"""
    address: str
    private_key: str
    account: LocalAccount


class A2ABlockchainClient:
    """
    Client for interacting with A2A Network smart contracts
    Handles agent registration, messaging, and trust verification
    """
    
    def __init__(
        self,
        rpc_url: str = None,
        config_manager: ContractConfigManager = None,
        private_key: str = None
    ):
        # Get RPC URL with validation
        self.rpc_url = rpc_url or os.getenv("A2A_RPC_URL")
        
        if not self.rpc_url:
            if os.getenv("NODE_ENV") == "production":
                raise ValueError("A2A_RPC_URL environment variable required in production")
            else:
                logger.warning("No A2A_RPC_URL configured, falling back to localhost (development only)")
                self.rpc_url = "http://localhost:8545"
        
        # SECURITY FIX: Validate RPC URL format and security
        self._validate_rpc_url(self.rpc_url)
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Verify connection
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {self.rpc_url}")
        
        logger.info(f"Connected to A2A Network at {self.rpc_url}")
        logger.info(f"Chain ID: {self.web3.eth.chain_id}")
        
        # Load or create agent identity
        self.agent_identity = self._load_or_create_agent_identity(private_key)
        
        # Load contract configuration
        self.config_manager = config_manager or get_contract_config()
        
        # Validate configuration
        validation = self.config_manager.validate_configuration()
        if not validation['valid']:
            logger.error("Contract configuration validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Invalid contract configuration")
        
        # Load smart contracts using dynamic configuration
        self.agent_registry_contract = self._load_contract_from_config("AgentRegistry")
        self.message_router_contract = self._load_contract_from_config("MessageRouter")
        self.ord_registry_contract = self._load_contract_from_config("ORDRegistry")
        
        # ENHANCEMENT: Initialize nonce and transaction managers
        self.nonce_manager = NonceManager(self.web3)
        self.transaction_manager = TransactionManager(self.web3, self.nonce_manager)
        
        logger.info(f"Agent identity: {self.agent_identity.address}")
        if self.config_manager.is_contract_available("AgentRegistry"):
            logger.info(f"AgentRegistry: {self.config_manager.get_contract_address('AgentRegistry')}")
        if self.config_manager.is_contract_available("MessageRouter"):
            logger.info(f"MessageRouter: {self.config_manager.get_contract_address('MessageRouter')}")
        if self.config_manager.is_contract_available("ORDRegistry"):
            logger.info(f"ORDRegistry: {self.config_manager.get_contract_address('ORDRegistry')}")
    
    def _load_or_create_agent_identity(self, private_key: str = None) -> AgentIdentity:
        """Load existing agent identity or create new one"""
        if private_key:
            account = Account.from_key(private_key)
        else:
            # Try to load from environment or create new
            env_key = os.getenv("A2A_AGENT_PRIVATE_KEY")
            if env_key:
                account = Account.from_key(env_key)
            else:
                # Production security: Never create temporary accounts
                if os.getenv("NODE_ENV") == "production":
                    raise ValueError("Private key required in production. Set A2A_AGENT_PRIVATE_KEY environment variable")
                
                # Development only: Create temporary account with strong warnings
                logger.error("SECURITY WARNING: Creating temporary account for development only")
                logger.error("This account will be lost when the process restarts")
                logger.error("Set A2A_AGENT_PRIVATE_KEY environment variable for persistent identity")
                
                # Validate we're actually in development
                if os.getenv("NODE_ENV") in ["staging", "production"]:
                    raise ValueError("Cannot create temporary accounts in staging/production environments")
                
                account = Account.create()
                logger.warning(f"Temporary agent identity created: {account.address}")
                # SECURITY FIX: Never log private keys
        
        return AgentIdentity(
            address=account.address,
            private_key=account.key.hex(),
            account=account
        )
    
    def _validate_rpc_url(self, rpc_url: str) -> None:
        """Validate RPC URL for security and format"""
        try:
            parsed = urlparse(rpc_url)
            
            # Check for valid scheme
            if parsed.scheme not in ['http', 'https', 'ws', 'wss']:
                raise ValueError(f"Invalid RPC URL scheme: {parsed.scheme}. Must be http, https, ws, or wss")
            
            # Enforce HTTPS/WSS in production
            if os.getenv("NODE_ENV") == "production":
                if parsed.scheme not in ['https', 'wss']:
                    raise ValueError("Production requires secure RPC connection (https or wss)")
                
                # Prevent localhost in production
                if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0', None]:
                    raise ValueError("Cannot use localhost RPC URL in production environment")
            
            # Validate hostname exists
            if not parsed.hostname:
                raise ValueError("RPC URL must include a valid hostname")
                
            # Check for suspicious ports
            if parsed.port and parsed.port not in [80, 443, 8545, 8546, 30303]:
                logger.warning(f"Non-standard port {parsed.port} used for RPC connection")
                
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid RPC URL format: {e}")
    
    def _load_contract_from_config(self, contract_name: str) -> Optional[Contract]:
        """Load smart contract using dynamic configuration"""
        try:
            contract_info = self.config_manager.get_contract(contract_name)
            
            if not contract_info:
                logger.warning(f"Contract {contract_name} not found in configuration")
                return self._create_minimal_contract(contract_name)
            
            if not contract_info.address or contract_info.address == '0x':
                logger.warning(f"Contract {contract_name} has invalid address: {contract_info.address}")
                return self._create_minimal_contract(contract_name)
            
            # Use ABI from configuration or fallback to minimal
            abi = contract_info.abi if contract_info.abi else self._get_minimal_abi(contract_name)
            
            # Convert to checksum address
            checksum_address = self.web3.to_checksum_address(contract_info.address)
            contract = self.web3.eth.contract(address=checksum_address, abi=abi)
            logger.info(f"Loaded {contract_name} contract from configuration: {checksum_address}")
            return contract
            
        except Exception as e:
            logger.error(f"Failed to load {contract_name} contract from configuration: {e}")
            # Create minimal contract for development
            return self._create_minimal_contract(contract_name)
    
    def _get_minimal_abi(self, contract_name: str) -> List[Dict]:
        """Minimal ABI for development when artifacts not available"""
        if contract_name == "AgentRegistry":
            return [
                {
                    "name": "registerAgent",
                    "type": "function",
                    "inputs": [
                        {"name": "name", "type": "string"},
                        {"name": "endpoint", "type": "string"},
                        {"name": "capabilities", "type": "bytes32[]"}
                    ],
                    "outputs": []
                },
                {
                    "name": "getAgent",
                    "type": "function",
                    "inputs": [{"name": "agentAddress", "type": "address"}],
                    "outputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "name", "type": "string"},
                        {"name": "endpoint", "type": "string"},
                        {"name": "capabilities", "type": "bytes32[]"},
                        {"name": "reputation", "type": "uint256"},
                        {"name": "active", "type": "bool"},
                        {"name": "registeredAt", "type": "uint256"}
                    ]
                }
            ]
        elif contract_name == "MessageRouter":
            return [
                {
                    "name": "sendMessage",
                    "type": "function",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "content", "type": "string"},
                        {"name": "messageType", "type": "bytes32"}
                    ],
                    "outputs": [{"name": "messageId", "type": "bytes32"}]
                },
                {
                    "name": "getMessage",
                    "type": "function",
                    "inputs": [{"name": "messageId", "type": "bytes32"}],
                    "outputs": [
                        {"name": "from", "type": "address"},
                        {"name": "to", "type": "address"},
                        {"name": "messageId", "type": "bytes32"},
                        {"name": "content", "type": "string"},
                        {"name": "timestamp", "type": "uint256"},
                        {"name": "delivered", "type": "bool"},
                        {"name": "messageType", "type": "bytes32"}
                    ]
                }
            ]
        return []
    
    def _create_minimal_contract(self, contract_name: str) -> Optional[Contract]:
        """Create minimal contract instance for development"""
        # In production, contracts must be properly configured - no fallbacks allowed
        if os.getenv("NODE_ENV") == "production":
            raise ValueError(f"Contract {contract_name} not properly configured for production environment. Set contract address in environment or deployment config.")
        
        # In staging, warn but don't create fallback contracts
        if os.getenv("NODE_ENV") == "staging":
            logger.error(f"Contract {contract_name} not configured in staging environment")
            return None
        
        # Development only: Return None instead of zero address contract
        logger.error(f"Contract {contract_name} not configured - functionality will be limited")
        logger.error("Configure proper contract addresses for full functionality")
        
        return None  # Return None instead of zero address contract
    
    async def register_agent(
        self,
        name: str,
        endpoint: str,
        capabilities: List[str]
    ) -> bool:
        """Register agent on blockchain"""
        try:
            # Convert capabilities to bytes32
            capability_hashes = [
                self.web3.keccak(text=cap)[:32] for cap in capabilities
            ]
            
            # ENHANCEMENT: Build transaction without nonce (managed by TransactionManager)
            transaction = self.agent_registry_contract.functions.registerAgent(
                name,
                endpoint,
                capability_hashes
            ).build_transaction({
                'from': self.agent_identity.address,
                'gas': int(os.getenv('BLOCKCHAIN_GAS_LIMIT', '500000')),
                'gasPrice': int(float(os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI', str(self.web3.eth.gas_price))) * 1e9) if os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI') else self.web3.eth.gas_price
            })
            
            # ENHANCEMENT: Use transaction manager for nonce handling and retry logic
            tx_hash = await self.transaction_manager.submit_transaction(
                transaction, 
                self.agent_identity.account
            )
            
            # Wait for confirmation with proper error handling
            receipt = await self.transaction_manager.wait_for_confirmation(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Agent registered successfully: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Agent registration failed: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False
    
    async def send_message(
        self,
        to_address: str,
        content: str,
        message_type: str = "a2a_message"
    ) -> Optional[str]:
        """Send message to another agent via blockchain"""
        try:
            message_type_hash = self.web3.keccak(text=message_type)[:32]
            
            # Build transaction
            transaction = self.message_router_contract.functions.sendMessage(
                to_address,
                content,
                message_type_hash
            ).build_transaction({
                'from': self.agent_identity.address,
                'nonce': self.web3.eth.get_transaction_count(self.agent_identity.address),
                'gas': int(os.getenv('BLOCKCHAIN_MESSAGE_GAS_LIMIT', '300000')),
                'gasPrice': int(float(os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI', str(self.web3.eth.gas_price))) * 1e9) if os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI') else self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_txn = self.agent_identity.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                # Extract message ID from logs
                message_id = receipt.logs[0].data.hex() if receipt.logs else tx_hash.hex()
                logger.info(f"Message sent successfully: {message_id}")
                return message_id
            else:
                logger.error(f"Message sending failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None
    
    async def get_agent_info(self, agent_address: str) -> Optional[Dict]:
        """Get agent information from blockchain"""
        try:
            result = self.agent_registry_contract.functions.getAgent(agent_address).call()
            
            return {
                "owner": result[0],
                "name": result[1],
                "endpoint": result[2],
                "capabilities": result[3],
                "reputation": result[4],
                "active": result[5],
                "registered_at": result[6]
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent info: {e}")
            return None
    
    async def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with specific capability"""
        try:
            capability_hash = self.web3.keccak(text=capability)[:32]
            agents = self.agent_registry_contract.functions.findAgentsByCapability(capability_hash).call()
            return agents
            
        except Exception as e:
            logger.error(f"Failed to find agents by capability: {e}")
            return []
    
    def get_balance(self) -> float:
        """Get agent's ETH balance"""
        try:
            balance_wei = self.web3.eth.get_balance(self.agent_identity.address)
            return self.web3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    async def is_agent_registered(self) -> bool:
        """Check if current agent is registered"""
        agent_info = await self.get_agent_info(self.agent_identity.address)
        return agent_info is not None and agent_info.get("active", False)
    
    async def register_ord_document(
        self,
        title: str,
        description: str,
        document_uri: str,
        capabilities: List[str],
        tags: List[str],
        dublin_core: Dict[str, str]
    ) -> Optional[str]:
        """Register ORD document on blockchain"""
        try:
            # Convert capabilities and tags to bytes32
            capability_hashes = [self.web3.keccak(text=cap)[:32] for cap in capabilities]
            tag_hashes = [self.web3.keccak(text=tag)[:32] for tag in tags]
            
            # Prepare Dublin Core metadata
            dc_metadata = (
                dublin_core.get('creator', ''),
                dublin_core.get('subject', ''),
                dublin_core.get('contributor', ''),
                dublin_core.get('publisher', ''),
                dublin_core.get('type', ''),
                dublin_core.get('format', ''),
                dublin_core.get('identifier', ''),
                dublin_core.get('source', ''),
                dublin_core.get('language', ''),
                dublin_core.get('relation', ''),
                dublin_core.get('coverage', ''),
                dublin_core.get('rights', '')
            )
            
            # Build transaction
            transaction = self.ord_registry_contract.functions.registerORDDocument(
                title,
                description,
                document_uri,
                capability_hashes,
                tag_hashes,
                dc_metadata
            ).build_transaction({
                'from': self.agent_identity.address,
                'nonce': self.web3.eth.get_transaction_count(self.agent_identity.address),
                'gas': int(os.getenv('BLOCKCHAIN_ORD_GAS_LIMIT', '800000')),
                'gasPrice': int(float(os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI', str(self.web3.eth.gas_price))) * 1e9) if os.getenv('BLOCKCHAIN_GAS_PRICE_GWEI') else self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_txn = self.agent_identity.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                # Extract document ID from logs
                document_id = receipt.logs[0].data.hex() if receipt.logs else tx_hash.hex()
                logger.info(f"ORD document registered: {document_id}")
                return document_id
            else:
                logger.error(f"ORD document registration failed: {tx_hash.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to register ORD document: {e}")
            return None
    
    async def find_ord_documents_by_capability(self, capability: str) -> List[str]:
        """Find ORD documents by capability"""
        try:
            capability_hash = self.web3.keccak(text=capability)[:32]
            publishers = self.ord_registry_contract.functions.findDocumentsByCapability(capability_hash).call()
            return publishers
        except Exception as e:
            logger.error(f"Failed to find ORD documents by capability: {e}")
            return []
    
    async def find_ord_documents_by_tag(self, tag: str) -> List[str]:
        """Find ORD documents by tag"""
        try:
            tag_hash = self.web3.keccak(text=tag)[:32]
            publishers = self.ord_registry_contract.functions.findDocumentsByTag(tag_hash).call()
            return publishers
        except Exception as e:
            logger.error(f"Failed to find ORD documents by tag: {e}")
            return []
    
    async def get_ord_document(self, document_id: str) -> Optional[Dict]:
        """Get ORD document details from blockchain"""
        try:
            result = self.ord_registry_contract.functions.getORDDocument(document_id).call()
            dublin_core = self.ord_registry_contract.functions.getDublinCoreMetadata(document_id).call()
            
            return {
                "document_id": result[0].hex(),
                "publisher": result[1],
                "title": result[2],
                "description": result[3],
                "document_uri": result[4],
                "capabilities": result[5],
                "tags": result[6],
                "version": result[7],
                "published_at": result[8],
                "updated_at": result[9],
                "active": result[10],
                "reputation": result[11],
                "dublin_core": {
                    "creator": dublin_core[0],
                    "subject": dublin_core[1],
                    "contributor": dublin_core[2],
                    "publisher": dublin_core[3],
                    "type": dublin_core[4],
                    "format": dublin_core[5],
                    "identifier": dublin_core[6],
                    "source": dublin_core[7],
                    "language": dublin_core[8],
                    "relation": dublin_core[9],
                    "coverage": dublin_core[10],
                    "rights": dublin_core[11]
                }
            }
        except Exception as e:
            logger.error(f"Failed to get ORD document: {e}")
            return None


# Global blockchain client instance
_blockchain_client: Optional[A2ABlockchainClient] = None

def get_blockchain_client() -> A2ABlockchainClient:
    """Get or create global blockchain client instance"""
    global _blockchain_client
    if _blockchain_client is None:
        _blockchain_client = A2ABlockchainClient()
    return _blockchain_client

def initialize_blockchain_client(
    rpc_url: str = None,
    config_manager: ContractConfigManager = None,
    private_key: str = None
) -> A2ABlockchainClient:
    """Initialize blockchain client with custom configuration"""
    global _blockchain_client
    _blockchain_client = A2ABlockchainClient(
        rpc_url=rpc_url,
        config_manager=config_manager,
        private_key=private_key
    )
    return _blockchain_client
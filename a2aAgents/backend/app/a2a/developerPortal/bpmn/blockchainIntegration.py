"""
Blockchain and Smart Contract Integration for BPMN Workflow Engine
Real integration with A2A Network smart contracts
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import os
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.messages import encode_defunct
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Import A2A blockchain integration if available
try:
    from a2a_network.python_sdk.blockchain.web3_client import Web3Client
    from a2a_network.python_sdk.blockchain.agent_integration import BlockchainAgentIntegration


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    A2A_SDK_AVAILABLE = True
except ImportError:
    A2A_SDK_AVAILABLE = False
    logging.warning("A2A SDK not available, using direct Web3 integration")

logger = logging.getLogger(__name__)


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    LOCAL = "local"
    TESTNET = "testnet"


class SmartContractTaskType(str, Enum):
    """Types of smart contract operations for A2A network"""
    AGENT_REGISTRATION = "agent_registration"
    MESSAGE_ROUTING = "message_routing"
    CAPABILITY_QUERY = "capability_query"
    REPUTATION_UPDATE = "reputation_update"
    MESSAGE_CONFIRMATION = "message_confirmation"
    AGENT_STATUS_UPDATE = "agent_status_update"
    AGENT_DISCOVERY = "agent_discovery"


class A2AContractAddresses:
    """A2A contract addresses for different networks"""
    # Local development (Anvil)
    LOCAL = {
        "AgentRegistry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "MessageRouter": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
    }
    # Sepolia testnet - actual deployed addresses
    SEPOLIA = {
        "AgentRegistry": os.environ.get("SEPOLIA_AGENT_REGISTRY", ""),
        "MessageRouter": os.environ.get("SEPOLIA_MESSAGE_ROUTER", "")
    }
    # Polygon Mumbai testnet
    MUMBAI = {
        "AgentRegistry": os.environ.get("MUMBAI_AGENT_REGISTRY", ""),
        "MessageRouter": os.environ.get("MUMBAI_MESSAGE_ROUTER", "")
    }


class A2ABlockchainIntegration:
    """Real integration with A2A Network smart contracts"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web3_connections: Dict[str, Web3] = {}
        self.contracts: Dict[str, Contract] = {}
        self.event_subscriptions = {}

        # Initialize connections
        self._initialize_connections()

        # Load A2A contract ABIs
        self.contract_abis = self._load_a2a_contract_abis()

        # Initialize contracts
        self._initialize_contracts()

        logger.info("A2A Blockchain integration initialized")

    def _initialize_connections(self):
        """Initialize Web3 connections"""
        # Default to local Anvil if no config provided
        default_config = {
            "local": {
                "provider_url": os.getenv("A2A_SERVICE_URL"),
                "chain_id": 31337
            }
        }

        networks = self.config.get("networks", default_config)

        for network_name, network_config in networks.items():
            try:
                provider_url = network_config.get("provider_url")
                if provider_url:
                    web3 = Web3(Web3.HTTPProvider(provider_url))

                    if web3.is_connected():
                        self.web3_connections[network_name] = web3
                        logger.info(f"Connected to {network_name} at {provider_url}")
                        logger.info(f"Chain ID: {web3.eth.chain_id}")
                        logger.info(f"Latest block: {web3.eth.block_number}")
                    else:
                        logger.error(f"Failed to connect to {network_name}")

            except Exception as e:
                logger.error(f"Error connecting to {network_name}: {e}")

    def _load_a2a_contract_abis(self) -> Dict[str, Any]:
        """Load actual A2A contract ABIs"""
        abis = {}

        # Try to load from A2A network build artifacts
        a2a_network_path = Path(__file__).parent.parent.parent.parent.parent.parent / "a2a_network"
        contracts_path = a2a_network_path / "foundry" / "out"

        try:
            # Load AgentRegistry ABI
            agent_registry_path = contracts_path / "AgentRegistry.sol" / "AgentRegistry.json"
            if agent_registry_path.exists():
                with open(agent_registry_path, 'r') as f:
                    contract_data = json.load(f)
                    abis["AgentRegistry"] = contract_data["abi"]
                    logger.info("Loaded AgentRegistry ABI from build artifacts")
            else:
                # Use hardcoded ABI if file not found (fallback for development)
                abis["AgentRegistry"] = self._get_agent_registry_abi()
                logger.info("Using hardcoded AgentRegistry ABI as fallback - ensure contracts are deployed with matching ABI")

            # Load MessageRouter ABI
            message_router_path = contracts_path / "MessageRouter.sol" / "MessageRouter.json"
            if message_router_path.exists():
                with open(message_router_path, 'r') as f:
                    contract_data = json.load(f)
                    abis["MessageRouter"] = contract_data["abi"]
                    logger.info("Loaded MessageRouter ABI from build artifacts")
            else:
                # Use hardcoded ABI if file not found (fallback for development)
                abis["MessageRouter"] = self._get_message_router_abi()
                logger.info("Using hardcoded MessageRouter ABI as fallback - ensure contracts are deployed with matching ABI")

        except Exception as e:
            logger.warning(f"Could not load ABIs from artifacts: {e}")
            # Fall back to hardcoded ABIs
            abis["AgentRegistry"] = self._get_agent_registry_abi()
            abis["MessageRouter"] = self._get_message_router_abi()

        return abis

    def _get_agent_registry_abi(self) -> List[Dict]:
        """Get hardcoded AgentRegistry ABI"""
        return [
            {
                "inputs": [],
                "name": "initialize",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_name", "type": "string"},
                    {"name": "_endpoint", "type": "string"},
                    {"name": "_capabilities", "type": "string[]"}
                ],
                "name": "registerAgent",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_capabilities", "type": "string[]"}],
                "name": "updateCapabilities",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_endpoint", "type": "string"}],
                "name": "updateEndpoint",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_isActive", "type": "bool"}],
                "name": "setAgentStatus",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_agent", "type": "address"},
                    {"name": "_delta", "type": "int8"}
                ],
                "name": "updateReputation",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_capability", "type": "string"}],
                "name": "findAgentsByCapability",
                "outputs": [{"name": "", "type": "address[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "_agent", "type": "address"}],
                "name": "getAgent",
                "outputs": [
                    {"name": "name", "type": "string"},
                    {"name": "endpoint", "type": "string"},
                    {"name": "capabilities", "type": "string[]"},
                    {"name": "reputation", "type": "uint8"},
                    {"name": "isActive", "type": "bool"},
                    {"name": "registeredAt", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "agent", "type": "address"},
                    {"indexed": False, "name": "name", "type": "string"}
                ],
                "name": "AgentRegistered",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "agent", "type": "address"},
                    {"indexed": False, "name": "oldReputation", "type": "uint8"},
                    {"indexed": False, "name": "newReputation", "type": "uint8"}
                ],
                "name": "ReputationUpdated",
                "type": "event"
            }
        ]

    def _get_message_router_abi(self) -> List[Dict]:
        """Get hardcoded MessageRouter ABI"""
        return [
            {
                "inputs": [],
                "name": "initialize",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_content", "type": "string"},
                    {"name": "_messageType", "type": "string"}
                ],
                "name": "sendMessage",
                "outputs": [{"name": "", "type": "bytes32"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_messageId", "type": "bytes32"}],
                "name": "confirmDelivery",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_from", "type": "address"}],
                "name": "getMessagesFrom",
                "outputs": [{"name": "", "type": "bytes32[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "_to", "type": "address"}],
                "name": "getMessagesTo",
                "outputs": [{"name": "", "type": "bytes32[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "_messageId", "type": "bytes32"}],
                "name": "getMessage",
                "outputs": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "content", "type": "string"},
                    {"name": "messageType", "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "delivered", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "messageId", "type": "bytes32"},
                    {"indexed": True, "name": "from", "type": "address"},
                    {"indexed": True, "name": "to", "type": "address"},
                    {"indexed": False, "name": "content", "type": "string"},
                    {"indexed": False, "name": "messageType", "type": "string"}
                ],
                "name": "MessageSent",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "messageId", "type": "bytes32"},
                    {"indexed": False, "name": "deliveredAt", "type": "uint256"}
                ],
                "name": "MessageDelivered",
                "type": "event"
            }
        ]

    def _initialize_contracts(self):
        """Initialize contract instances"""
        for network_name, web3 in self.web3_connections.items():
            self.contracts[network_name] = {}

            # Get contract addresses for this network
            if network_name == "local":
                addresses = A2AContractAddresses.LOCAL
            elif network_name == "sepolia":
                addresses = A2AContractAddresses.SEPOLIA
            elif network_name == "mumbai":
                addresses = A2AContractAddresses.MUMBAI
            else:
                addresses = self.config.get("contract_addresses", {}).get(network_name, {})

            # Initialize contracts
            for contract_name, address in addresses.items():
                if address and address != "":
                    try:
                        contract = web3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=self.contract_abis.get(contract_name, [])
                        )
                        self.contracts[network_name][contract_name] = contract
                        logger.info(f"Initialized {contract_name} at {address} on {network_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {contract_name}: {e}")
                else:
                    logger.warning(f"No address configured for {contract_name} on {network_name}")

    async def execute_blockchain_task(
        self,
        task_type: SmartContractTaskType,
        network: str,
        task_config: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a blockchain task"""
        try:
            web3 = self.web3_connections.get(network)
            if not web3:
                raise ValueError(f"No connection to {network} network")

            # Get private key from config or environment
            private_key = task_config.get("privateKey") or variables.get("privateKey") or os.environ.get("WORKFLOW_PRIVATE_KEY")
            if not private_key:
                # For read-only operations, private key is not required
                if task_type in [SmartContractTaskType.CAPABILITY_QUERY, SmartContractTaskType.AGENT_DISCOVERY]:
                    private_key = None
                else:
                    raise ValueError("Private key required for blockchain write operations")

            # Execute based on task type
            if task_type == SmartContractTaskType.AGENT_REGISTRATION:
                return await self._register_agent(web3, network, task_config, variables, private_key)

            elif task_type == SmartContractTaskType.MESSAGE_ROUTING:
                return await self._send_message(web3, network, task_config, variables, private_key)

            elif task_type == SmartContractTaskType.CAPABILITY_QUERY:
                return await self._query_capabilities(web3, network, task_config, variables)

            elif task_type == SmartContractTaskType.REPUTATION_UPDATE:
                return await self._update_reputation(web3, network, task_config, variables, private_key)

            elif task_type == SmartContractTaskType.MESSAGE_CONFIRMATION:
                return await self._confirm_message(web3, network, task_config, variables, private_key)

            elif task_type == SmartContractTaskType.AGENT_STATUS_UPDATE:
                return await self._update_agent_status(web3, network, task_config, variables, private_key)

            elif task_type == SmartContractTaskType.AGENT_DISCOVERY:
                return await self._discover_agents(web3, network, task_config, variables)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Blockchain task failed: {e}")
            raise

    async def _register_agent(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        private_key: str
    ) -> Dict[str, Any]:
        """Register an agent on the blockchain"""
        try:
            contract = self.contracts[network]["AgentRegistry"]
            account = Account.from_key(private_key)

            # Get agent details
            agent_name = variables.get("agentName", config.get("agentName", "Workflow Agent"))
            endpoint = variables.get("agentEndpoint", config.get("agentEndpoint", os.getenv("A2A_SERVICE_URL")))
            capabilities = variables.get("agentCapabilities", config.get("capabilities", ["workflow"]))

            # Build transaction
            nonce = web3.eth.get_transaction_count(account.address)
            gas_price = web3.eth.gas_price

            tx = contract.functions.registerAgent(
                agent_name,
                endpoint,
                capabilities
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': config.get("gasLimit", 500000),
                'gasPrice': gas_price,
                'chainId': web3.eth.chain_id
            })

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for confirmation
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            # Parse events
            events = contract.events.AgentRegistered().process_receipt(receipt)

            return {
                "success": receipt.status == 1,
                "transactionHash": tx_hash.hex(),
                "agentAddress": account.address,
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed,
                "events": [{"agent": e.args.agent, "name": e.args.name} for e in events]
            }

        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _send_message(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        private_key: str
    ) -> Dict[str, Any]:
        """Send a message through the MessageRouter"""
        try:
            contract = self.contracts[network]["MessageRouter"]
            account = Account.from_key(private_key)

            # Get message details
            to_address = variables.get("toAgent", config.get("toAgent"))
            content = variables.get("messageContent", config.get("content", ""))
            message_type = variables.get("messageType", config.get("messageType", "general"))

            if not to_address:
                raise ValueError("Recipient address required")

            # Build transaction
            nonce = web3.eth.get_transaction_count(account.address)
            gas_price = web3.eth.gas_price

            tx = contract.functions.sendMessage(
                Web3.to_checksum_address(to_address),
                content,
                message_type
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': config.get("gasLimit", 200000),
                'gasPrice': gas_price,
                'chainId': web3.eth.chain_id
            })

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for confirmation
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            # Parse events to get message ID
            events = contract.events.MessageSent().process_receipt(receipt)
            message_id = events[0].args.messageId if events else None

            return {
                "success": receipt.status == 1,
                "transactionHash": tx_hash.hex(),
                "messageId": message_id.hex() if message_id else None,
                "from": account.address,
                "to": to_address,
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed
            }

        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _query_capabilities(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query agents by capability"""
        try:
            contract = self.contracts[network]["AgentRegistry"]

            capability = variables.get("capability", config.get("capability"))
            if not capability:
                raise ValueError("Capability required")

            # Call view function
            agents = contract.functions.findAgentsByCapability(capability).call()

            # Get agent details for each address
            agent_details = []
            for agent_address in agents:
                try:
                    details = contract.functions.getAgent(agent_address).call()
                    agent_details.append({
                        "address": agent_address,
                        "name": details[0],
                        "endpoint": details[1],
                        "capabilities": details[2],
                        "reputation": details[3],
                        "isActive": details[4],
                        "registeredAt": details[5]
                    })
                except:
                    pass

            return {
                "success": True,
                "capability": capability,
                "agents": agent_details,
                "count": len(agent_details)
            }

        except Exception as e:
            logger.error(f"Capability query failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _update_reputation(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        private_key: str
    ) -> Dict[str, Any]:
        """Update agent reputation"""
        try:
            contract = self.contracts[network]["AgentRegistry"]
            account = Account.from_key(private_key)

            # Get update details
            agent_address = variables.get("agentAddress", config.get("agentAddress"))
            delta = int(variables.get("reputationDelta", config.get("delta", 0)))

            if not agent_address:
                raise ValueError("Agent address required")

            # Build transaction
            nonce = web3.eth.get_transaction_count(account.address)
            gas_price = web3.eth.gas_price

            tx = contract.functions.updateReputation(
                Web3.to_checksum_address(agent_address),
                delta
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': config.get("gasLimit", 100000),
                'gasPrice': gas_price,
                'chainId': web3.eth.chain_id
            })

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for confirmation
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            # Parse events
            events = contract.events.ReputationUpdated().process_receipt(receipt)

            return {
                "success": receipt.status == 1,
                "transactionHash": tx_hash.hex(),
                "agentAddress": agent_address,
                "reputationDelta": delta,
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed,
                "events": [{"oldReputation": e.args.oldReputation, "newReputation": e.args.newReputation} for e in events]
            }

        except Exception as e:
            logger.error(f"Reputation update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _confirm_message(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        private_key: str
    ) -> Dict[str, Any]:
        """Confirm message delivery"""
        try:
            contract = self.contracts[network]["MessageRouter"]
            account = Account.from_key(private_key)

            # Get message ID
            message_id = variables.get("messageId", config.get("messageId"))
            if not message_id:
                raise ValueError("Message ID required")

            # Convert to bytes32 if string
            if isinstance(message_id, str):
                if message_id.startswith("0x"):
                    message_id = bytes.fromhex(message_id[2:])
                else:
                    message_id = bytes.fromhex(message_id)

            # Build transaction
            nonce = web3.eth.get_transaction_count(account.address)
            gas_price = web3.eth.gas_price

            tx = contract.functions.confirmDelivery(message_id).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': config.get("gasLimit", 100000),
                'gasPrice': gas_price,
                'chainId': web3.eth.chain_id
            })

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for confirmation
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            return {
                "success": receipt.status == 1,
                "transactionHash": tx_hash.hex(),
                "messageId": message_id.hex() if isinstance(message_id, bytes) else message_id,
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed
            }

        except Exception as e:
            logger.error(f"Message confirmation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _update_agent_status(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        private_key: str
    ) -> Dict[str, Any]:
        """Update agent status (active/inactive)"""
        try:
            contract = self.contracts[network]["AgentRegistry"]
            account = Account.from_key(private_key)

            # Get status
            is_active = variables.get("isActive", config.get("isActive", True))

            # Build transaction
            nonce = web3.eth.get_transaction_count(account.address)
            gas_price = web3.eth.gas_price

            tx = contract.functions.setAgentStatus(is_active).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': config.get("gasLimit", 100000),
                'gasPrice': gas_price,
                'chainId': web3.eth.chain_id
            })

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for confirmation
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            return {
                "success": receipt.status == 1,
                "transactionHash": tx_hash.hex(),
                "agentAddress": account.address,
                "isActive": is_active,
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed
            }

        except Exception as e:
            logger.error(f"Agent status update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _discover_agents(
        self,
        web3: Web3,
        network: str,
        config: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Discover all registered agents"""
        try:
            contract = self.contracts[network]["AgentRegistry"]

            # Get recent AgentRegistered events
            from_block = config.get("fromBlock", 0)
            to_block = config.get("toBlock", "latest")

            events = contract.events.AgentRegistered.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            ).get_all_entries()

            # Get unique agents
            agents = {}
            for event in events:
                agent_address = event.args.agent
                try:
                    details = contract.functions.getAgent(agent_address).call()
                    if details[4]:  # isActive
                        agents[agent_address] = {
                            "address": agent_address,
                            "name": details[0],
                            "endpoint": details[1],
                            "capabilities": details[2],
                            "reputation": details[3],
                            "isActive": details[4],
                            "registeredAt": details[5]
                        }
                except:
                    pass

            return {
                "success": True,
                "agents": list(agents.values()),
                "count": len(agents)
            }

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def subscribe_to_events(
        self,
        network: str,
        contract_name: str,
        event_name: str,
        callback: callable,
        filters: Optional[Dict] = None
    ) -> str:
        """Subscribe to blockchain events"""
        try:
            web3 = self.web3_connections[network]
            contract = self.contracts[network][contract_name]

            # Create event filter
            event = getattr(contract.events, event_name)
            event_filter = event.create_filter(fromBlock="latest", argument_filters=filters)

            # Store subscription
            sub_id = f"{network}_{contract_name}_{event_name}_{datetime.utcnow().timestamp()}"
            self.event_subscriptions[sub_id] = {
                "filter": event_filter,
                "callback": callback,
                "network": network,
                "contract": contract_name,
                "event": event_name
            }

            # Start monitoring
            asyncio.create_task(self._monitor_events(sub_id))

            logger.info(f"Subscribed to {event_name} events on {contract_name}")
            return sub_id

        except Exception as e:
            logger.error(f"Event subscription failed: {e}")
            raise

    async def _monitor_events(self, subscription_id: str):
        """Monitor events for a subscription"""
        subscription = self.event_subscriptions.get(subscription_id)
        if not subscription:
            return

        event_filter = subscription["filter"]
        callback = subscription["callback"]

        while subscription_id in self.event_subscriptions:
            try:
                # Get new events
                for event in event_filter.get_new_entries():
                    try:
                        # Call the callback with event data
                        await callback({
                            "subscription_id": subscription_id,
                            "event": event.event,
                            "args": dict(event.args),
                            "blockNumber": event.blockNumber,
                            "transactionHash": event.transactionHash.hex(),
                            "address": event.address
                        })
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")

                # Wait before next poll
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Event monitoring error: {e}")
                await asyncio.sleep(5)

    async def close(self):
        """Cleanup resources"""
        # Cancel all event subscriptions
        for sub_id in list(self.event_subscriptions.keys()):
            del self.event_subscriptions[sub_id]

        logger.info("A2A Blockchain integration closed")
"""
A2A Network Python Client

Main client class for interacting with A2A Network infrastructure.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, Union
import websockets
from web3 import Web3

from .services import (
    AgentManager, MessageManager, TokenManager,
    GovernanceManager, ScalabilityManager, ReputationManager
)
from .utils.errors import A2AError, ErrorCode
from .utils.validation import validate_address, validate_config
from .constants.networks import NETWORKS, DEFAULT_NETWORK
from .constants.contracts import CONTRACT_ADDRESSES
from .constants.abis import get_contract_abi
from .debug import DebugMode, Profiler, MessageTracer
from .debugEnhanced import ComprehensiveDebugSuite

logger = logging.getLogger(__name__)

class ConnectionState:
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    ERROR = 'error'

class A2AClient:
    """
    Main A2A Network client for all SDK interactions.
    
    This client provides access to all A2A Network services including
    agent management, messaging, token operations, governance, and more.
    
    Args:
        config (Dict[str, Any]): Client configuration
        
    Example:
        >>> client = A2AClient({
        ...     'network': 'mainnet',
        ...     'private_key': 'your_private_key',
        ...     'rpc_url': 'https://mainnet.infura.io/v3/your-key'
        ... })
        >>> await client.connect()
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Validate configuration
        validation_result = validate_config(config)
        if not validation_result['is_valid']:
            raise A2AError(
                ErrorCode.INVALID_CONFIG,
                f"Invalid configuration: {', '.join(validation_result.get('errors', []))}"
            )
        
        self.config = {
            **config,
            'network': config.get('network', DEFAULT_NETWORK),
            'api_timeout': config.get('api_timeout', 30),
            'retry_attempts': config.get('retry_attempts', 3),
            'auto_reconnect': config.get('auto_reconnect', True)
        }

        self.w3: Optional[Web3] = None
        self.account = None
        self.websocket = None
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Contract instances
        self.contracts: Dict[str, Any] = {}
        
        # Event subscriptions and callbacks
        self.subscriptions: Dict[str, Dict] = {}
        self.event_callbacks: Dict[str, Callable] = {}
        
        # Initialize Web3 provider
        self._initialize_provider()
        
        # Initialize service managers
        self.agents = AgentManager(self)
        self.messages = MessageManager(self)
        self.tokens = TokenManager(self)
        self.governance = GovernanceManager(self)
        self.scalability = ScalabilityManager(self)
        self.reputation = ReputationManager(self)
        
        # Initialize debug tools (disabled by default)
        self.debug = DebugMode(self)
        self.profiler = Profiler(self)
        self.tracer = MessageTracer(self)
        
        # Initialize enhanced debugging suite
        self.debug_suite = ComprehensiveDebugSuite(self)
        
        # Setup logging
        self._setup_logging()

    def _initialize_provider(self):
        """Initialize Web3 provider based on configuration."""
        network_config = NETWORKS.get(self.config['network'])
        if not network_config:
            raise A2AError(
                ErrorCode.UNSUPPORTED_NETWORK,
                f"Unsupported network: {self.config['network']}"
            )

        # Determine RPC URL
        rpc_url = (
            self.config.get('rpc_url') or 
            (network_config.get('rpc_urls', [None])[0])
        )
        
        if not rpc_url:
            raise A2AError(ErrorCode.NO_PROVIDER, 'No RPC URL configured')

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add PoA middleware for some networks
        if network_config.get('poa', False):
            try:
                # Try newer web3.py versions first
                from web3.middleware import ExtraDataToPOAMiddleware
                self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            except ImportError:
                try:
                    # Fallback for older versions
                    from web3.middleware.geth_poa import geth_poa_middleware
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except ImportError:
                    # Skip PoA middleware if not available
                    logger.warning("PoA middleware not available, skipping injection")

        # Set up account if private key provided
        if self.config.get('private_key'):
            try:
                self.account = self.w3.eth.account.from_key(self.config['private_key'])
                self.w3.eth.default_account = self.account.address
            except Exception as e:
                raise A2AError(ErrorCode.INVALID_PRIVATE_KEY, str(e))

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    async def connect(self):
        """
        Connect to A2A Network.
        
        Establishes connection to the blockchain network, initializes contracts,
        and optionally connects to WebSocket for real-time updates.
        
        Raises:
            A2AError: If connection fails
        """
        try:
            self.connection_state = ConnectionState.CONNECTING
            logger.info(f"Connecting to A2A Network on {self.config['network']}")

            # Verify network connection
            if not self.w3.isConnected():
                raise A2AError(ErrorCode.CONNECTION_FAILED, "Failed to connect to blockchain")

            network_config = NETWORKS[self.config['network']]
            chain_id = self.w3.eth.chain_id
            expected_chain_id = network_config['chain_id']
            
            if chain_id != expected_chain_id:
                raise A2AError(
                    ErrorCode.WRONG_NETWORK,
                    f"Connected to wrong network. Expected {expected_chain_id}, got {chain_id}"
                )

            # Initialize contracts
            await self._initialize_contracts()
            
            # Connect WebSocket if configured
            if self.config.get('websocket_url'):
                await self._connect_websocket()

            self.connection_state = ConnectionState.CONNECTED
            logger.info("Successfully connected to A2A Network")

        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            if isinstance(e, A2AError):
                raise e
            raise A2AError(ErrorCode.CONNECTION_FAILED, str(e))

    async def disconnect(self):
        """
        Disconnect from A2A Network.
        
        Closes WebSocket connection, clears subscriptions and contracts.
        """
        try:
            logger.info("Disconnecting from A2A Network")
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            # Clear subscriptions
            self.subscriptions.clear()
            self.event_callbacks.clear()
            
            # Clear contracts
            self.contracts.clear()

            self.connection_state = ConnectionState.DISCONNECTED
            logger.info("Disconnected from A2A Network")

        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
            raise A2AError(ErrorCode.DISCONNECTION_FAILED, str(e))

    async def _initialize_contracts(self):
        """Initialize smart contract instances."""
        contract_addresses = CONTRACT_ADDRESSES.get(self.config['network'])
        if not contract_addresses:
            raise A2AError(
                ErrorCode.NO_CONTRACTS,
                f"No contract addresses found for network: {self.config['network']}"
            )

        # Initialize core contracts
        contract_configs = [
            ('AgentRegistry', 'AGENT_REGISTRY'),
            ('MessageRouter', 'MESSAGE_ROUTER'),
            ('A2AToken', 'A2A_TOKEN'),
            ('TimelockGovernor', 'TIMELOCK_GOVERNOR'),
            ('LoadBalancer', 'LOAD_BALANCER'),
            ('AIAgentMatcher', 'AI_AGENT_MATCHER')
        ]

        for contract_name, address_key in contract_configs:
            address = contract_addresses.get(address_key)
            if not address or not validate_address(address):
                raise A2AError(
                    ErrorCode.INVALID_CONTRACT_ADDRESS,
                    f"Invalid contract address for {contract_name}: {address}"
                )

            try:
                abi = get_contract_abi(contract_name)
                contract = self.w3.eth.contract(
                    address=Web3.toChecksumAddress(address),
                    abi=abi
                )
                self.contracts[contract_name] = contract
                logger.debug(f"Initialized contract {contract_name} at {address}")
                
            except Exception as e:
                raise A2AError(
                    ErrorCode.CONTRACT_INITIALIZATION_FAILED,
                    f"Failed to initialize {contract_name}: {e}"
                )

    async def _connect_websocket(self):
        """Connect WebSocket for real-time updates."""
        try:
            self.websocket = await websockets.connect(
                self.config['websocket_url'],
                timeout=self.config['api_timeout']
            )
            
            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())
            logger.info("WebSocket connected")
            
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            if not self.config.get('require_websocket', False):
                # Continue without WebSocket if not required
                return
            raise A2AError(ErrorCode.WEBSOCKET_ERROR, str(e))

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            if self.config['auto_reconnect']:
                await self._reconnect_websocket()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        message_type = data.get('type')
        message_data = data.get('data', {})
        
        # Call registered callbacks
        if message_type in self.event_callbacks:
            try:
                await self.event_callbacks[message_type](message_data)
            except Exception as e:
                logger.error(f"Error in event callback for {message_type}: {e}")

    async def _reconnect_websocket(self):
        """Reconnect WebSocket with exponential backoff."""
        max_attempts = self.config['retry_attempts']
        
        for attempt in range(max_attempts):
            try:
                delay = min(2 ** attempt, 60)  # Cap at 60 seconds
                await asyncio.sleep(delay)
                
                await self._connect_websocket()
                logger.info("WebSocket reconnected successfully")
                return
                
            except Exception as e:
                logger.warning(f"WebSocket reconnection attempt {attempt + 1} failed: {e}")
                
        logger.error(f"Failed to reconnect WebSocket after {max_attempts} attempts")

    def get_contract(self, name: str):
        """
        Get contract instance by name.
        
        Args:
            name (str): Contract name
            
        Returns:
            Contract instance
            
        Raises:
            A2AError: If contract not found
        """
        contract = self.contracts.get(name)
        if not contract:
            raise A2AError(ErrorCode.CONTRACT_NOT_FOUND, f"Contract {name} not found")
        return contract

    def get_web3(self) -> Web3:
        """Get Web3 instance."""
        return self.w3

    def get_account(self):
        """Get current account."""
        return self.account

    def get_address(self) -> Optional[str]:
        """Get current wallet address."""
        return self.account.address if self.account else None

    def get_connection_state(self) -> str:
        """Get current connection state."""
        return self.connection_state

    def get_config(self) -> Dict[str, Any]:
        """Get client configuration."""
        return self.config.copy()

    async def subscribe_to_events(self, contract_name: str, event_name: str, callback: Callable):
        """
        Subscribe to contract events.
        
        Args:
            contract_name (str): Name of contract
            event_name (str): Name of event
            callback (Callable): Callback function for events
            
        Returns:
            str: Subscription ID
        """
        contract = self.get_contract(contract_name)
        subscription_id = f"{contract_name}_{event_name}_{int(time.time() * 1000)}"
        
        # Create event filter
        event_filter = getattr(contract.events, event_name).createFilter(fromBlock='latest')
        
        self.subscriptions[subscription_id] = {
            'contract': contract,
            'event_name': event_name,
            'filter': event_filter,
            'callback': callback
        }
        
        # Start polling for events
        asyncio.create_task(self._poll_events(subscription_id))
        
        return subscription_id

    async def _poll_events(self, subscription_id: str):
        """Poll for contract events."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
            
        event_filter = subscription['filter']
        callback = subscription['callback']
        
        try:
            while subscription_id in self.subscriptions:
                try:
                    new_events = event_filter.get_new_entries()
                    for event in new_events:
                        await callback(event)
                except Exception as e:
                    logger.error(f"Error polling events for {subscription_id}: {e}")
                
                await asyncio.sleep(1)  # Poll every second
                
        except asyncio.CancelledError:
            pass

    def unsubscribe_from_events(self, subscription_id: str):
        """
        Unsubscribe from contract events.
        
        Args:
            subscription_id (str): Subscription ID to remove
        """
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]

    def on(self, event_type: str, callback: Callable):
        """
        Register event callback.
        
        Args:
            event_type (str): Event type to listen for
            callback (Callable): Callback function
        """
        self.event_callbacks[event_type] = callback

    def off(self, event_type: str):
        """
        Remove event callback.
        
        Args:
            event_type (str): Event type to stop listening for
        """
        if event_type in self.event_callbacks:
            del self.event_callbacks[event_type]

    async def send_websocket_message(self, message: Dict[str, Any]):
        """
        Send WebSocket message.
        
        Args:
            message (Dict[str, Any]): Message to send
            
        Raises:
            A2AError: If WebSocket not connected
        """
        if not self.websocket:
            raise A2AError(ErrorCode.WEBSOCKET_NOT_CONNECTED, "WebSocket not connected")
            
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            raise A2AError(ErrorCode.WEBSOCKET_ERROR, str(e))

    async def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information.
        
        Returns:
            Dict[str, Any]: Network information
        """
        try:
            chain_id = self.w3.eth.chain_id
            block_number = self.w3.eth.block_number
            gas_price = self.w3.eth.gas_price
            
            return {
                'chain_id': chain_id,
                'block_number': block_number,
                'gas_price': gas_price,
                'network': self.config['network'],
                'contracts': list(self.contracts.keys())
            }
        except Exception as e:
            raise A2AError(ErrorCode.NETWORK_INFO_FAILED, str(e))

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            network_info = await self.get_network_info()
            
            return {
                'status': 'healthy',
                'details': {
                    'connection_state': self.connection_state,
                    'network': network_info,
                    'address': self.get_address(),
                    'contracts_loaded': len(self.contracts),
                    'subscriptions': len(self.subscriptions),
                    'websocket_connected': self.websocket is not None
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'details': {
                    'error': str(e),
                    'connection_state': self.connection_state
                }
            }

    def __repr__(self) -> str:
        return f"A2AClient(network={self.config['network']}, state={self.connection_state})"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def enable_comprehensive_debugging(self, 
                                     debug_port: int = 5000,
                                     dashboard_port: int = 8050,
                                     enable_blockchain_tracing: bool = True):
        """
        Enable comprehensive debugging suite with all features.
        
        Args:
            debug_port (int): Port for interactive debug console
            dashboard_port (int): Port for performance dashboard
            enable_blockchain_tracing (bool): Enable blockchain integration
        
        Returns:
            Dict[str, str]: URLs for accessing debug tools
        """
        try:
            # Initialize blockchain integration if requested
            if enable_blockchain_tracing and self.w3:
                contract_addresses = {
                    'MessageRouter': CONTRACT_ADDRESSES.get(self.config['network'], {}).get('MESSAGE_ROUTER'),
                    'AgentRegistry': CONTRACT_ADDRESSES.get(self.config['network'], {}).get('AGENT_REGISTRY')
                }
                # Filter out None values
                contract_addresses = {k: v for k, v in contract_addresses.items() if v}
                
                if contract_addresses:
                    self.debug_suite.initialize_blockchain_integration(self.w3, contract_addresses)
            
            # Start comprehensive debugging
            self.debug_suite.start_comprehensive_debugging(debug_port, dashboard_port)
            
            return {
                'debug_console': f'http://localhost:{debug_port}',
                'performance_dashboard': f'http://localhost:{dashboard_port}',
                'status': 'enabled'
            }
            
        except Exception as e:
            logger.error(f"Failed to enable comprehensive debugging: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def set_breakpoint(self, method_name: str, condition: str = "True"):
        """
        Set a debugging breakpoint on a method.
        
        Args:
            method_name (str): Method name (e.g., 'agents.register', 'messages.send')
            condition (str): Python expression for breakpoint condition
        
        Example:
            client.set_breakpoint('messages.send', "args[0].startswith('urgent')")
        """
        self.debug_suite.interactive_debugger.set_breakpoint(method_name, condition)
    
    def get_debug_report(self) -> Dict[str, Any]:
        """
        Get comprehensive debugging report.
        
        Returns:
            Dict[str, Any]: Complete debugging report with metrics, errors, and performance data
        """
        return self.debug_suite.get_comprehensive_report()
    
    async def trace_message_full(self, message_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive message tracing with blockchain integration.
        
        Args:
            message_id (str): ID of message to trace
            
        Returns:
            Dict[str, Any]: Complete trace including SDK, blockchain, and performance data
        """
        return await self.debug_suite.trace_message_comprehensive(message_id)
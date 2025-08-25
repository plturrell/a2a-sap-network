"""
Blockchain Event Streaming Service
Replaces WebSocket connections with blockchain event monitoring for real-time updates
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Set
from datetime import datetime
from collections import defaultdict

from web3 import Web3
from eth_utils import is_address

logger = logging.getLogger(__name__)


class BlockchainEventStream:
    """
    Real-time event streaming using blockchain events instead of WebSockets
    Provides A2A protocol compliant real-time updates
    """

    def __init__(self, blockchain_url: str = None, contract_address: str = None):
        self.blockchain_url = blockchain_url or "http://localhost:8545"
        self.contract_address = contract_address

        # Web3 connection
        self.w3 = None
        self.contract = None

        # Event subscriptions
        self.event_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # event_type -> Set[subscriber_id]
        self.subscriber_callbacks: Dict[str, Callable] = {}  # subscriber_id -> callback
        self.event_filters = {}  # event_type -> filter

        # Event queue for processing
        self.event_queue = asyncio.Queue()
        self.processing_task = None

        # Metrics
        self.metrics = {
            'events_processed': 0,
            'events_missed': 0,
            'subscribers_active': 0,
            'last_block': 0
        }

        self._initialize_blockchain()

    def _initialize_blockchain(self):
        """Initialize blockchain connection"""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.blockchain_url))

            if not self.w3.isConnected():
                logger.error("Failed to connect to blockchain")
                return

            logger.info(f"Connected to blockchain at {self.blockchain_url}")

            # Load contract ABI (simplified for example)
            contract_abi = [
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "from", "type": "address"},
                        {"indexed": True, "name": "to", "type": "address"},
                        {"indexed": False, "name": "messageType", "type": "string"},
                        {"indexed": False, "name": "data", "type": "string"},
                        {"indexed": False, "name": "timestamp", "type": "uint256"}
                    ],
                    "name": "A2AMessage",
                    "type": "event"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "agentId", "type": "string"},
                        {"indexed": False, "name": "eventType", "type": "string"},
                        {"indexed": False, "name": "data", "type": "string"},
                        {"indexed": False, "name": "timestamp", "type": "uint256"}
                    ],
                    "name": "AgentEvent",
                    "type": "event"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "metricType", "type": "string"},
                        {"indexed": False, "name": "value", "type": "uint256"},
                        {"indexed": False, "name": "metadata", "type": "string"},
                        {"indexed": False, "name": "timestamp", "type": "uint256"}
                    ],
                    "name": "MetricUpdate",
                    "type": "event"
                }
            ]

            if self.contract_address and is_address(self.contract_address):
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=contract_abi
                )
                logger.info(f"Loaded contract at {self.contract_address}")

        except Exception as e:
            logger.error(f"Blockchain initialization failed: {e}")

    async def start(self):
        """Start the event streaming service"""
        if not self.w3 or not self.w3.isConnected():
            logger.error("Cannot start: blockchain not connected")
            return

        # Start event processing
        self.processing_task = asyncio.create_task(self._process_events())

        # Start blockchain monitoring
        asyncio.create_task(self._monitor_blockchain())

        logger.info("Blockchain event streaming started")

    async def stop(self):
        """Stop the event streaming service"""
        if self.processing_task:
            self.processing_task.cancel()

        # Clear all subscriptions
        self.event_subscriptions.clear()
        self.subscriber_callbacks.clear()
        self.event_filters.clear()

        logger.info("Blockchain event streaming stopped")

    def subscribe(self, subscriber_id: str, event_types: List[str], callback: Callable):
        """
        Subscribe to blockchain events

        Args:
            subscriber_id: Unique identifier for the subscriber
            event_types: List of event types to subscribe to
            callback: Async function to call when events occur
        """
        # Store callback
        self.subscriber_callbacks[subscriber_id] = callback

        # Add to subscriptions
        for event_type in event_types:
            self.event_subscriptions[event_type].add(subscriber_id)

            # Create event filter if needed
            if event_type not in self.event_filters and self.contract:
                self._create_event_filter(event_type)

        self.metrics['subscribers_active'] = len(self.subscriber_callbacks)

        logger.info(f"Subscriber {subscriber_id} subscribed to events: {event_types}")

    def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from all events"""
        # Remove from all event subscriptions
        for subscribers in self.event_subscriptions.values():
            subscribers.discard(subscriber_id)

        # Remove callback
        self.subscriber_callbacks.pop(subscriber_id, None)

        self.metrics['subscribers_active'] = len(self.subscriber_callbacks)

        logger.info(f"Subscriber {subscriber_id} unsubscribed")

    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """
        Publish an event to the blockchain

        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            if not self.w3 or not self.contract:
                logger.error("Cannot publish: blockchain not initialized")
                return

            # In production, this would send a transaction to emit the event
            # For now, we'll simulate by adding to queue
            event = {
                'type': event_type,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'block_number': self.w3.eth.block_number
            }

            await self.event_queue.put(event)

            logger.debug(f"Published event: {event_type}")

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    def _create_event_filter(self, event_type: str):
        """Create blockchain event filter"""
        try:
            if not self.contract:
                return

            # Map event types to contract events
            event_mapping = {
                'a2a_message': self.contract.events.A2AMessage,
                'agent_event': self.contract.events.AgentEvent,
                'metric_update': self.contract.events.MetricUpdate
            }

            if event_type in event_mapping:
                # Create filter from latest block
                event_filter = event_mapping[event_type].createFilter(
                    fromBlock='latest'
                )
                self.event_filters[event_type] = event_filter

                logger.info(f"Created event filter for: {event_type}")

        except Exception as e:
            logger.error(f"Failed to create event filter: {e}")

    async def _monitor_blockchain(self):
        """Monitor blockchain for new events"""
        while True:
            try:
                if not self.w3 or not self.w3.isConnected():
                    await asyncio.sleep(5)
                    continue

                # Get latest block
                latest_block = self.w3.eth.block_number

                if latest_block > self.metrics['last_block']:
                    # Check all event filters
                    for event_type, event_filter in self.event_filters.items():
                        try:
                            # Get new events
                            for event in event_filter.get_new_entries():
                                await self._handle_blockchain_event(event_type, event)
                        except Exception as e:
                            logger.error(f"Error processing events for {event_type}: {e}")

                    self.metrics['last_block'] = latest_block

                # Check every second
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Blockchain monitoring error: {e}")
                await asyncio.sleep(5)

    async def _handle_blockchain_event(self, event_type: str, event):
        """Handle event from blockchain"""
        try:
            # Parse event data
            event_data = {
                'type': event_type,
                'data': dict(event['args']) if hasattr(event, 'args') else event,
                'block_number': event.get('blockNumber'),
                'transaction_hash': event.get('transactionHash').hex() if event.get('transactionHash') else None,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Add to processing queue
            await self.event_queue.put(event_data)

        except Exception as e:
            logger.error(f"Failed to handle blockchain event: {e}")
            self.metrics['events_missed'] += 1

    async def _process_events(self):
        """Process events from queue and notify subscribers"""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                event_type = event['type']

                # Get subscribers for this event type
                subscribers = self.event_subscriptions.get(event_type, set())

                # Notify all subscribers
                for subscriber_id in subscribers:
                    callback = self.subscriber_callbacks.get(subscriber_id)
                    if callback:
                        try:
                            await callback(event)
                        except Exception as e:
                            logger.error(f"Subscriber {subscriber_id} callback error: {e}")

                self.metrics['events_processed'] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        return {
            **self.metrics,
            'connected': self.w3.isConnected() if self.w3 else False,
            'queue_size': self.event_queue.qsize(),
            'event_types': list(self.event_filters.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }


class A2AEventStreamAdapter:
    """
    Adapter to replace WebSocket functionality with blockchain events
    Provides WebSocket-like API using blockchain
    """

    def __init__(self, blockchain_stream: BlockchainEventStream):
        self.stream = blockchain_stream
        self.connections = {}  # connection_id -> subscription_info

    async def handle_connection(self, connection_id: str, send_callback: Callable):
        """Handle new connection (replaces WebSocket connection)"""
        # Store connection info
        self.connections[connection_id] = {
            'id': connection_id,
            'send': send_callback,
            'subscriptions': set()
        }

        # Send initial connection message
        await send_callback({
            'type': 'connection',
            'status': 'connected',
            'connection_id': connection_id,
            'timestamp': datetime.utcnow().isoformat()
        })

        logger.info(f"Blockchain stream connection established: {connection_id}")

    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle message from client (replaces WebSocket message)"""
        if connection_id not in self.connections:
            logger.error(f"Unknown connection: {connection_id}")
            return

        conn_info = self.connections[connection_id]
        action = message.get('action')

        if action == 'subscribe':
            # Subscribe to events
            event_types = message.get('events', [])

            # Create callback for this connection
            async def event_callback(event):
                await conn_info['send']({
                    'type': 'event',
                    'event': event,
                    'timestamp': datetime.utcnow().isoformat()
                })

            # Subscribe to blockchain events
            self.stream.subscribe(connection_id, event_types, event_callback)
            conn_info['subscriptions'].update(event_types)

            # Send confirmation
            await conn_info['send']({
                'type': 'subscription_confirmed',
                'events': event_types,
                'timestamp': datetime.utcnow().isoformat()
            })

        elif action == 'unsubscribe':
            # Unsubscribe from events
            self.stream.unsubscribe(connection_id)
            conn_info['subscriptions'].clear()

            await conn_info['send']({
                'type': 'unsubscribed',
                'timestamp': datetime.utcnow().isoformat()
            })

        elif action == 'publish':
            # Publish event
            event_type = message.get('event_type')
            event_data = message.get('data', {})

            if event_type:
                await self.stream.publish_event(event_type, event_data)

                await conn_info['send']({
                    'type': 'published',
                    'event_type': event_type,
                    'timestamp': datetime.utcnow().isoformat()
                })

    async def handle_disconnect(self, connection_id: str):
        """Handle disconnection"""
        if connection_id in self.connections:
            # Unsubscribe from all events
            self.stream.unsubscribe(connection_id)

            # Remove connection
            del self.connections[connection_id]

            logger.info(f"Blockchain stream connection closed: {connection_id}")

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.connections)


# Example usage for replacing WebSocket server
async def create_blockchain_event_server(port: int = 4006):
    """
    Create blockchain event streaming server to replace WebSocket server
    """
    # Initialize blockchain stream
    blockchain_stream = BlockchainEventStream(
        blockchain_url="http://localhost:8545",
        contract_address="0x..."  # A2A contract address
    )

    # Start streaming
    await blockchain_stream.start()

    # Create adapter
    adapter = A2AEventStreamAdapter(blockchain_stream)

    # In production, this would integrate with the HTTP server
    # to handle upgrade requests and convert them to blockchain streaming

    logger.info(f"Blockchain event streaming server ready (replaces WebSocket on port {port})")

    return blockchain_stream, adapter


# Singleton instance
_stream_instance = None


def get_blockchain_stream() -> BlockchainEventStream:
    """Get singleton blockchain stream instance"""
    global _stream_instance
    if not _stream_instance:
        _stream_instance = BlockchainEventStream()
    return _stream_instance
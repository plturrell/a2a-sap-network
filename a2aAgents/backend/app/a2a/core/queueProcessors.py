"""
Queue Processor Implementations for Enhanced Queue Manager
Provides reusable processors for common queue patterns
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



import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
# Direct HTTP calls not allowed - use A2A protocol
# import aiohttp  # REMOVED: A2A protocol violation
from .queueManager import QueueProcessor, QueueMessage, QueuePriority

logger = logging.getLogger(__name__)


class BatchProcessor(QueueProcessor):
    """Process messages in batches for efficiency"""

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 5.0,
        processor_func: Optional[Callable] = None,
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.processor_func = processor_func
        self._batch: List[QueueMessage] = []
        self._batch_lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._processing_task = None

    async def process(self, message: QueueMessage) -> bool:
        """Add message to batch"""
        async with self._batch_lock:
            self._batch.append(message)

            if len(self._batch) >= self.batch_size:
                # Process batch immediately if full
                return await self._process_batch()
            else:
                # Start timeout if this is the first message
                if len(self._batch) == 1:
                    self._processing_task = asyncio.create_task(self._batch_timeout_handler())
                return True

    async def _batch_timeout_handler(self):
        """Process batch after timeout"""
        await asyncio.sleep(self.batch_timeout)
        async with self._batch_lock:
            if self._batch:
                await self._process_batch()

    async def _process_batch(self) -> bool:
        """Process accumulated batch"""
        if not self._batch:
            return True

        batch_to_process = self._batch.copy()
        self._batch.clear()

        try:
            if self.processor_func:
                # Process with custom function
                payloads = [msg.payload for msg in batch_to_process]
                results = await self.processor_func(payloads)
                return all(results) if isinstance(results, list) else results
            else:
                logger.info(f"Batch processing {len(batch_to_process)} messages")
                return True
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return False


class HTTPProcessor(QueueProcessor):
    """Process messages by sending to HTTP endpoints"""

    def __init__(
        self,
        endpoint_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        retry_on_failure: bool = True,
    ):
        self.endpoint_url = endpoint_url
        self.method = method
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure

    async def process(self, message: QueueMessage) -> bool:
        """Send message to HTTP endpoint"""
        try:
            async with # WARNING: aiohttp ClientSession usage violates A2A protocol - must use blockchain messaging
        # aiohttp\.ClientSession() as session:
                request_data = {
                    "message_id": message.id,
                    "payload": message.payload,
                    "metadata": message.metadata,
                    "timestamp": message.created_at.isoformat(),
                }

                async with session.request(
                    method=self.method,
                    url=self.endpoint_url,
                    json=request_data,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.debug(
                            f"Successfully sent message {message.id} to {self.endpoint_url}"
                        )
                        return True
                    else:
                        logger.error(f"HTTP error {response.status} for message {message.id}")
                        return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout sending message {message.id} to {self.endpoint_url}")
            return False
        except Exception as e:
            logger.error(f"Error sending message {message.id}: {e}")
            return False


class TransformProcessor(QueueProcessor):
    """Transform messages before forwarding to another queue"""

    def __init__(
        self,
        transform_func: Callable[[Any], Any],
        target_queue: Optional["EnhancedQueue"] = None,
        filter_func: Optional[Callable[[Any], bool]] = None,
    ):
        self.transform_func = transform_func
        self.target_queue = target_queue
        self.filter_func = filter_func

    async def process(self, message: QueueMessage) -> bool:
        """Transform and optionally forward message"""
        try:
            # Apply filter if provided
            if self.filter_func and not self.filter_func(message.payload):
                logger.debug(f"Message {message.id} filtered out")
                return True

            # Transform payload
            if asyncio.iscoroutinefunction(self.transform_func):
                transformed_payload = await self.transform_func(message.payload)
            else:
                transformed_payload = self.transform_func(message.payload)

            # Forward to target queue if provided
            if self.target_queue:
                await self.target_queue.enqueue(
                    payload=transformed_payload,
                    priority=message.priority,
                    metadata={
                        **message.metadata,
                        "original_message_id": message.id,
                        "transformed_at": datetime.utcnow().isoformat(),
                    },
                )

            return True

        except Exception as e:
            logger.error(f"Transform processing failed for message {message.id}: {e}")
            return False


class AnalyticsProcessor(QueueProcessor):
    """Collect analytics on message flow"""

    def __init__(self, metrics_store: Optional[Dict[str, Any]] = None):
        self.metrics = metrics_store or {}
        self._message_types = {}
        self._processing_times = []
        self._priority_counts = {p: 0 for p in QueuePriority}

    async def process(self, message: QueueMessage) -> bool:
        """Collect analytics on message"""
        try:
            # Track message type
            msg_type = message.metadata.get("type", "unknown")
            self._message_types[msg_type] = self._message_types.get(msg_type, 0) + 1

            # Track priority distribution
            self._priority_counts[message.priority] += 1

            # Track processing delay
            processing_delay = (datetime.utcnow() - message.created_at).total_seconds()
            self._processing_times.append(processing_delay)

            # Update shared metrics if provided
            if self.metrics is not None:
                self.metrics.update(
                    {
                        "total_messages": sum(self._message_types.values()),
                        "message_types": self._message_types.copy(),
                        "priority_distribution": {
                            p.name: count for p, count in self._priority_counts.items()
                        },
                        "average_delay": (
                            sum(self._processing_times) / len(self._processing_times)
                            if self._processing_times
                            else 0
                        ),
                        "last_updated": datetime.utcnow().isoformat(),
                    }
                )

            return True

        except Exception as e:
            logger.error(f"Analytics processing failed: {e}")
            return True  # Don't fail the message


class RouterProcessor(QueueProcessor):
    """Route messages to different queues based on rules"""

    def __init__(self, routing_rules: Dict[str, Callable[[QueueMessage], Optional[str]]]):
        """
        Initialize with routing rules
        routing_rules: Dict mapping rule names to functions that return target queue name
        """
        self.routing_rules = routing_rules
        self._route_counts = {rule: 0 for rule in routing_rules}

    def add_rule(self, name: str, rule_func: Callable[[QueueMessage], Optional[str]]):
        """Add a routing rule"""
        self.routing_rules[name] = rule_func
        self._route_counts[name] = 0

    async def process(self, message: QueueMessage) -> bool:
        """Route message based on rules"""
        from .queueManager import queue_manager

        try:
            # Evaluate routing rules in order
            for rule_name, rule_func in self.routing_rules.items():
                target_queue_name = rule_func(message)

                if target_queue_name:
                    target_queue = queue_manager.get_queue(target_queue_name)

                    if target_queue:
                        await target_queue.enqueue(
                            payload=message.payload,
                            priority=message.priority,
                            metadata={
                                **message.metadata,
                                "routed_by": rule_name,
                                "routed_at": datetime.utcnow().isoformat(),
                            },
                        )
                        self._route_counts[rule_name] += 1
                        return True
                    else:
                        logger.warning(f"Target queue {target_queue_name} not found")

            # No routing rule matched
            logger.warning(f"No routing rule matched for message {message.id}")
            return False

        except Exception as e:
            logger.error(f"Routing failed for message {message.id}: {e}")
            return False


class ValidationProcessor(QueueProcessor):
    """Validate messages against schemas"""

    def __init__(
        self,
        schema: Dict[str, Any],
        validation_func: Optional[Callable[[Any], bool]] = None,
        fix_func: Optional[Callable[[Any], Any]] = None,
    ):
        self.schema = schema
        self.validation_func = validation_func
        self.fix_func = fix_func
        self._validation_stats = {"valid": 0, "invalid": 0, "fixed": 0}

    async def process(self, message: QueueMessage) -> bool:
        """Validate message payload"""
        try:
            # Custom validation function
            if self.validation_func:
                is_valid = self.validation_func(message.payload)
            else:
                # Basic schema validation
                is_valid = self._validate_schema(message.payload)

            if is_valid:
                self._validation_stats["valid"] += 1
                return True
            elif self.fix_func:
                # Try to fix invalid message
                fixed_payload = self.fix_func(message.payload)
                message.payload = fixed_payload
                self._validation_stats["fixed"] += 1
                return True
            else:
                self._validation_stats["invalid"] += 1
                logger.error(f"Message {message.id} failed validation")
                return False

        except Exception as e:
            logger.error(f"Validation error for message {message.id}: {e}")
            return False

    def _validate_schema(self, payload: Any) -> bool:
        """Basic schema validation"""
        if not isinstance(payload, dict):
            return False

        for field, field_type in self.schema.items():
            if field not in payload:
                return False
            if not isinstance(payload[field], field_type):
                return False

        return True


class ChainProcessor(QueueProcessor):
    """Chain multiple processors together"""

    def __init__(self, processors: List[QueueProcessor]):
        self.processors = processors

    async def process(self, message: QueueMessage) -> bool:
        """Process message through chain of processors"""
        for processor in self.processors:
            try:
                result = await processor.process(message)
                if not result:
                    logger.debug(f"Chain processing stopped at {processor.__class__.__name__}")
                    return False
            except Exception as e:
                logger.error(f"Chain processor error: {e}")
                return False

        return True


class ConditionalProcessor(QueueProcessor):
    """Process messages conditionally based on predicates"""

    def __init__(
        self,
        condition: Callable[[QueueMessage], bool],
        true_processor: QueueProcessor,
        false_processor: Optional[QueueProcessor] = None,
    ):
        self.condition = condition
        self.true_processor = true_processor
        self.false_processor = false_processor

    async def process(self, message: QueueMessage) -> bool:
        """Process based on condition"""
        try:
            if self.condition(message):
                return await self.true_processor.process(message)
            elif self.false_processor:
                return await self.false_processor.process(message)
            else:
                return True

        except Exception as e:
            logger.error(f"Conditional processing error: {e}")
            return False

"""
Enhanced Queue Management System for A2A Agents
Provides unified queue management with advanced features
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import heapq
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class QueueType(str, Enum):
    """Types of queues available"""

    PRIORITY = "priority"
    FIFO = "fifo"
    LIFO = "lifo"
    DELAYED = "delayed"
    BROADCAST = "broadcast"
    PARTITIONED = "partitioned"


class QueuePriority(int, Enum):
    """Queue priority levels"""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BULK = 4


class ProcessingStatus(str, Enum):
    """Message processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueMessage:
    """Enhanced queue message with metadata"""

    id: str
    payload: Any
    priority: QueuePriority = QueuePriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    partition_key: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def __lt__(self, other):
        """Enable priority queue sorting"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


@dataclass
class QueueMetrics:
    """Queue performance metrics"""

    messages_enqueued: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_expired: int = 0
    processing_time_total: float = 0.0
    queue_depth_current: int = 0
    queue_depth_max: int = 0
    last_activity: Optional[datetime] = None

    def record_enqueue(self):
        """Record message enqueue"""
        self.messages_enqueued += 1
        self.queue_depth_current += 1
        self.queue_depth_max = max(self.queue_depth_max, self.queue_depth_current)
        self.last_activity = datetime.utcnow()

    def record_processing(self, duration: float, success: bool):
        """Record message processing"""
        if success:
            self.messages_processed += 1
        else:
            self.messages_failed += 1
        self.processing_time_total += duration
        self.queue_depth_current = max(0, self.queue_depth_current - 1)
        self.last_activity = datetime.utcnow()


class QueueProcessor(ABC):
    """Abstract base class for queue processors"""

    @abstractmethod
    async def process(self, message: QueueMessage) -> bool:
        """Process a message. Return True if successful."""


class EnhancedQueue:
    """Enhanced queue implementation with advanced features"""

    def __init__(
        self,
        name: str,
        queue_type: QueueType = QueueType.PRIORITY,
        max_size: Optional[int] = None,
        ttl: Optional[timedelta] = None,
        dead_letter_queue: Optional["EnhancedQueue"] = None,
    ):
        self.name = name
        self.queue_type = queue_type
        self.max_size = max_size
        self.ttl = ttl
        self.dead_letter_queue = dead_letter_queue

        # Queue storage
        self._messages: List[QueueMessage] = []
        self._delayed_messages: List[Tuple[datetime, QueueMessage]] = []
        self._partitions: Dict[str, List[QueueMessage]] = defaultdict(list)

        # Metrics
        self.metrics = QueueMetrics()

        # Processors
        self._processors: List[QueueProcessor] = []
        self._processing_tasks: Set[asyncio.Task] = set()

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Processing control
        self._processing_enabled = True
        self._processing_lock = asyncio.Lock()

    async def enqueue(
        self,
        payload: Any,
        priority: QueuePriority = QueuePriority.MEDIUM,
        delay: Optional[timedelta] = None,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Enqueue a message"""
        # Check queue size limit
        if self.max_size and self.metrics.queue_depth_current >= self.max_size:
            raise ValueError(f"Queue {self.name} is full (max_size={self.max_size})")

        # Create message
        message_id = hashlib.md5(
            f"{time.time()}{json.dumps(payload, sort_keys=True)}".encode()
        ).hexdigest()

        scheduled_at = datetime.utcnow() + delay if delay else None
        expires_at = datetime.utcnow() + self.ttl if self.ttl else None

        message = QueueMessage(
            id=message_id,
            payload=payload,
            priority=priority,
            scheduled_at=scheduled_at,
            expires_at=expires_at,
            partition_key=partition_key,
            **kwargs,
        )

        # Store message based on type
        if delay:
            heapq.heappush(self._delayed_messages, (scheduled_at, message))
        elif partition_key and self.queue_type == QueueType.PARTITIONED:
            self._partitions[partition_key].append(message)
        else:
            if self.queue_type == QueueType.PRIORITY:
                heapq.heappush(self._messages, message)
            elif self.queue_type == QueueType.LIFO:
                self._messages.append(message)
            else:  # FIFO or default
                self._messages.append(message)

        # Update metrics
        self.metrics.record_enqueue()

        # Emit event
        await self._emit_event("message_enqueued", message)

        # Trigger processing
        if self._processing_enabled and self._processors:
            asyncio.create_task(self._process_next())

        return message_id

    async def dequeue(self, partition_key: Optional[str] = None) -> Optional[QueueMessage]:
        """Dequeue a message"""
        async with self._processing_lock:
            # Process delayed messages first
            await self._process_delayed_messages()

            # Remove expired messages
            await self._remove_expired_messages()

            # Get next message based on queue type
            message = None

            if partition_key and self.queue_type == QueueType.PARTITIONED:
                if partition_key in self._partitions and self._partitions[partition_key]:
                    message = self._partitions[partition_key].pop(0)
            elif self._messages:
                if self.queue_type == QueueType.PRIORITY:
                    message = heapq.heappop(self._messages)
                elif self.queue_type == QueueType.LIFO:
                    message = self._messages.pop()
                else:  # FIFO or default
                    message = self._messages.pop(0)

            if message:
                await self._emit_event("message_dequeued", message)

            return message

    async def _process_delayed_messages(self):
        """Move delayed messages to main queue if ready"""
        now = datetime.utcnow()
        ready_messages = []

        while self._delayed_messages and self._delayed_messages[0][0] <= now:
            _, message = heapq.heappop(self._delayed_messages)
            ready_messages.append(message)

        for message in ready_messages:
            if self.queue_type == QueueType.PRIORITY:
                heapq.heappush(self._messages, message)
            else:
                self._messages.append(message)

    async def _remove_expired_messages(self):
        """Remove expired messages"""
        now = datetime.utcnow()

        # Filter main queue
        self._messages = [
            msg for msg in self._messages if not msg.expires_at or msg.expires_at > now
        ]

        # Filter partitions
        for partition in self._partitions.values():
            partition[:] = [msg for msg in partition if not msg.expires_at or msg.expires_at > now]

        # Re-heapify if using priority queue
        if self.queue_type == QueueType.PRIORITY:
            heapq.heapify(self._messages)

    def add_processor(self, processor: QueueProcessor):
        """Add a message processor"""
        self._processors.append(processor)

    async def _process_next(self):
        """Process next available message"""
        if not self._processing_enabled or not self._processors:
            return

        message = await self.dequeue()
        if not message:
            return

        # Process with all processors
        start_time = time.time()
        success = True

        for processor in self._processors:
            try:
                result = await processor.process(message)
                if not result:
                    success = False
                    break
            except Exception as e:
                logger.error(f"Processor error in queue {self.name}: {e}")
                success = False
                break

        # Record metrics
        duration = time.time() - start_time
        self.metrics.record_processing(duration, success)

        # Handle failure
        if not success:
            await self._handle_failed_message(message)
        else:
            await self._emit_event("message_processed", message)

    async def _handle_failed_message(self, message: QueueMessage):
        """Handle failed message with retry or dead letter"""
        message.retry_count += 1

        if message.retry_count <= message.max_retries:
            # Retry with exponential backoff
            delay = timedelta(seconds=2**message.retry_count)
            message.scheduled_at = datetime.utcnow() + delay
            heapq.heappush(self._delayed_messages, (message.scheduled_at, message))
            await self._emit_event("message_retrying", message)
        elif self.dead_letter_queue:
            # Send to dead letter queue
            await self.dead_letter_queue.enqueue(
                payload=message.payload,
                priority=message.priority,
                metadata={
                    **message.metadata,
                    "original_queue": self.name,
                    "failure_count": message.retry_count,
                    "failed_at": datetime.utcnow().isoformat(),
                },
            )
            await self._emit_event("message_dead_lettered", message)
        else:
            # No more retries and no dead letter queue
            await self._emit_event("message_dropped", message)

    def on(self, event: str, handler: Callable):
        """Register event handler"""
        self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, data: Any):
        """Emit event to handlers"""
        for handler in self._event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error for {event}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        return {
            "name": self.name,
            "type": self.queue_type.value,
            "messages_enqueued": self.metrics.messages_enqueued,
            "messages_processed": self.metrics.messages_processed,
            "messages_failed": self.metrics.messages_failed,
            "messages_expired": self.metrics.messages_expired,
            "queue_depth": self.metrics.queue_depth_current,
            "queue_depth_max": self.metrics.queue_depth_max,
            "average_processing_time": (
                self.metrics.processing_time_total / self.metrics.messages_processed
                if self.metrics.messages_processed > 0
                else 0
            ),
            "last_activity": (
                self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            ),
        }

    async def start_processing(self):
        """Start message processing"""
        self._processing_enabled = True

        # Start continuous processing
        while self._processing_enabled:
            await self._process_next()
            await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning

    async def stop_processing(self):
        """Stop message processing"""
        self._processing_enabled = False

        # Wait for current processing to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)


class QueueManager:
    """Centralized queue manager for all agents"""

    def __init__(self):
        self._queues: Dict[str, EnhancedQueue] = {}
        self._global_metrics: Dict[str, Any] = defaultdict(int)
        self._queue_groups: Dict[str, Set[str]] = defaultdict(set)

    def create_queue(
        self,
        name: str,
        queue_type: QueueType = QueueType.PRIORITY,
        group: Optional[str] = None,
        **kwargs,
    ) -> EnhancedQueue:
        """Create a new queue"""
        if name in self._queues:
            raise ValueError(f"Queue {name} already exists")

        queue = EnhancedQueue(name, queue_type, **kwargs)
        self._queues[name] = queue

        if group:
            self._queue_groups[group].add(name)

        logger.info(f"Created queue {name} of type {queue_type}")
        return queue

    def get_queue(self, name: str) -> Optional[EnhancedQueue]:
        """Get queue by name"""
        return self._queues.get(name)

    def list_queues(self, group: Optional[str] = None) -> List[str]:
        """List all queues or queues in a group"""
        if group:
            return list(self._queue_groups.get(group, []))
        return list(self._queues.keys())

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all queues"""
        total_enqueued = 0
        total_processed = 0
        total_failed = 0
        total_depth = 0

        queue_metrics = {}
        for name, queue in self._queues.items():
            metrics = queue.get_metrics()
            queue_metrics[name] = metrics
            total_enqueued += metrics["messages_enqueued"]
            total_processed += metrics["messages_processed"]
            total_failed += metrics["messages_failed"]
            total_depth += metrics["queue_depth"]

        return {
            "total_queues": len(self._queues),
            "total_messages_enqueued": total_enqueued,
            "total_messages_processed": total_processed,
            "total_messages_failed": total_failed,
            "total_queue_depth": total_depth,
            "queue_metrics": queue_metrics,
        }

    async def broadcast(
        self, payload: Any, group: Optional[str] = None, exclude: Optional[Set[str]] = None
    ):
        """Broadcast message to multiple queues"""
        target_queues = (
            self._queue_groups.get(group, self._queues.keys()) if group else self._queues.keys()
        )
        exclude = exclude or set()

        tasks = []
        for queue_name in target_queues:
            if queue_name not in exclude and queue_name in self._queues:
                queue = self._queues[queue_name]
                tasks.append(queue.enqueue(payload, priority=QueuePriority.HIGH))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Singleton instance
queue_manager = QueueManager()

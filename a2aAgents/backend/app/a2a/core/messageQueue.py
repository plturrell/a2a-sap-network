"""
A2A Agent Message Queue System
Independent message queue with streaming and batch processing capabilities for each agent
No shared dependencies - each agent has its own queue instance
Enhanced with message serialization, validation, and compression
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from datetime import datetime, timedelta
from uuid import uuid4


def _generate_message_id() -> str:
    """Generate a unique message ID"""
    return str(uuid4())
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import heapq
from collections import deque
import threading
import time

from .messageSerialization import get_message_serializer, SerializationFormat, SerializedMessage
from .messageValidation import get_message_validator, ValidationResult, MessageValidationError

logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QueuedMessageStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ProcessingMode(str, Enum):
    IMMEDIATE = "immediate"  # Process instantly (streaming)
    QUEUED = "queued"  # Add to queue for batch processing
    AUTO = "auto"  # Decide based on current load


class QueuedMessage(BaseModel):
    """Message in the queue with metadata and serialization support"""

    message_id: str = Field(default_factory=_generate_message_id)
    a2a_message: Dict[str, Any]  # The actual A2A message
    context_id: str
    priority: MessagePriority = MessagePriority.MEDIUM
    status: QueuedMessageStatus = QueuedMessageStatus.QUEUED
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300  # 5 minutes default
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    # Serialization metadata
    serialized_message: Optional[SerializedMessage] = None
    serialization_format: Optional[SerializationFormat] = None
    message_size_bytes: int = 0
    compression_ratio: Optional[float] = None

    # Validation metadata
    validation_result: Optional[ValidationResult] = None
    validation_passed: bool = True
    validation_warnings: List[str] = Field(default_factory=list)

    @field_serializer("queued_at", "started_at", "completed_at")
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    model_config = ConfigDict()


class MessageProcessorStats(BaseModel):
    """Statistics for message processing"""

    total_processed: int = 0
    total_queued: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    avg_processing_time: float = 0.0
    queue_depth: int = 0
    concurrent_processing: int = 0
    last_reset: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("last_reset")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


class AgentMessageQueue:
    """Independent message queue for A2A agents with streaming and batch processing"""

    def __init__(
        self,
        agent_id: str,
        max_concurrent_processing: int = 5,
        auto_mode_threshold: int = 10,
        enable_streaming: bool = True,
        enable_batch_processing: bool = True,
        enable_message_compression: bool = True,
        compression_threshold: int = 1024,
        enable_message_validation: bool = True,
        strict_validation: bool = False,
    ):
        self.agent_id = agent_id
        self.max_concurrent_processing = max_concurrent_processing
        self.auto_mode_threshold = auto_mode_threshold
        self.enable_streaming = enable_streaming
        self.enable_batch_processing = enable_batch_processing
        self.enable_message_compression = enable_message_compression
        self.compression_threshold = compression_threshold
        self.enable_message_validation = enable_message_validation
        self.strict_validation = strict_validation

        # Priority queue using heapq (lower number = higher priority)
        self._priority_weights = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.MEDIUM: 2,
            MessagePriority.LOW: 3,
        }

        # Thread-safe message storage
        self._queue_lock = threading.RLock()
        self._priority_queue = []  # heap of (priority_weight, timestamp, message_id)
        self._messages: Dict[str, QueuedMessage] = {}  # message storage
        self._processing: Dict[str, QueuedMessage] = {}  # currently processing
        self._completed: deque = deque(maxlen=1000)  # completed messages (limited history)

        # Processing control
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = MessageProcessorStats()

        # Message processor callback (set by agent)
        self.message_processor: Optional[Callable] = None

        # Streaming subscribers
        self._streaming_subscribers: Dict[str, asyncio.Queue] = {}

        # Message serializer and validator
        self.message_serializer = get_message_serializer()
        self.message_validator = get_message_validator(strict_mode=strict_validation)

        logger.info(
            f"✅ Enhanced message queue initialized for agent {agent_id} "
            f"(compression: {enable_message_compression}, validation: {enable_message_validation})"
        )

    def set_message_processor(self, processor_func: Callable):
        """Set the message processing function (agent's process_message method)"""
        self.message_processor = processor_func

    async def start_queue_processor(self):
        """Start the background queue processor"""
        if not self.enable_batch_processing:
            return

        if self._queue_processor_task and not self._queue_processor_task.done():
            return

        self._queue_processor_task = asyncio.create_task(self._queue_processor_loop())
        logger.info(f"🚀 Queue processor started for agent {self.agent_id}")

    async def stop_queue_processor(self):
        """Stop the background queue processor"""
        self._shutdown_event.set()

        if self._queue_processor_task:
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass

        # Cancel all processing tasks
        for task in list(self._processing_tasks.values()):
            if not task.done():
                task.cancel()

        logger.info(f"⏹️ Queue processor stopped for agent {self.agent_id}")

    async def enqueue_message(
        self,
        a2a_message: Dict[str, Any],
        context_id: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
        processing_mode: ProcessingMode = ProcessingMode.AUTO,
        timeout_seconds: int = 300,
    ) -> str:
        """Add message to queue or process immediately with serialization optimization"""

        # Message validation, serialization, and optimization
        validation_result = None
        validation_passed = True
        validation_warnings = []
        serialized_message = None
        serialization_format = None
        message_size_bytes = 0
        compression_ratio = None

        # Convert dict to A2AMessage object for processing
        try:
            from .a2aTypes import A2AMessage

            a2a_msg_obj = (
                A2AMessage(**a2a_message) if isinstance(a2a_message, dict) else a2a_message
            )
        except Exception as e:
            logger.error(f"Failed to convert message to A2AMessage: {e}")
            raise ValueError(f"Invalid message format: {e}")

        # Message validation
        if self.enable_message_validation:
            try:
                validation_result = self.message_validator.validate_message(a2a_msg_obj)
                validation_passed = validation_result.is_valid
                validation_warnings = [issue.message for issue in validation_result.get_warnings()]

                logger.debug(
                    f"Message {a2a_msg_obj.messageId} validation: "
                    f"{'PASSED' if validation_passed else 'FAILED'}, "
                    f"{len(validation_result.get_errors())} errors, "
                    f"{len(validation_warnings)} warnings"
                )

                # Reject message if validation fails
                if not validation_passed:
                    error_messages = [issue.message for issue in validation_result.get_errors()]
                    raise MessageValidationError(
                        f"Message validation failed: {'; '.join(error_messages)}"
                    )

            except Exception as e:
                logger.error(f"Message validation error: {e}")
                if self.strict_validation:
                    raise
                else:
                    validation_passed = False
                    validation_warnings.append(f"Validation error: {str(e)}")

        # Message serialization for storage efficiency
        if self.enable_message_compression:
            try:
                # Determine optimal format
                optimal_format = self.message_serializer.get_optimal_format(a2a_msg_obj)

                # Serialize message
                serialized_message = self.message_serializer.serialize(a2a_msg_obj, optimal_format)
                serialization_format = serialized_message.format
                message_size_bytes = serialized_message.size_bytes
                compression_ratio = serialized_message.compression_ratio

                logger.debug(
                    f"Message {a2a_msg_obj.messageId} serialized: "
                    f"{message_size_bytes} bytes, format: {serialization_format}"
                )

            except Exception as e:
                logger.warning(f"Message serialization failed, using raw format: {e}")

        # Create queued message with serialization metadata
        queued_msg = QueuedMessage(
            a2a_message=a2a_message,
            context_id=context_id,
            priority=priority,
            timeout_seconds=timeout_seconds,
            serialized_message=serialized_message,
            serialization_format=serialization_format,
            message_size_bytes=message_size_bytes,
            compression_ratio=compression_ratio,
        )

        # Decide processing mode
        if processing_mode == ProcessingMode.AUTO:
            current_load = len(self._processing) + len(self._messages)
            if current_load >= self.auto_mode_threshold or not self.enable_streaming:
                processing_mode = ProcessingMode.QUEUED
            else:
                processing_mode = ProcessingMode.IMMEDIATE

        # Process based on mode
        if processing_mode == ProcessingMode.IMMEDIATE and self.enable_streaming:
            # Stream processing - process immediately
            logger.info(f"📡 Streaming message {queued_msg.message_id} for agent {self.agent_id}")
            asyncio.create_task(self._process_message_immediate(queued_msg))

        else:
            # Queue processing - add to queue
            with self._queue_lock:
                self._messages[queued_msg.message_id] = queued_msg
                priority_weight = self._priority_weights[priority]
                timestamp = time.time()
                heapq.heappush(
                    self._priority_queue, (priority_weight, timestamp, queued_msg.message_id)
                )
                self.stats.total_queued += 1
                self.stats.queue_depth = len(self._messages)

            logger.info(
                f"📥 Queued message {queued_msg.message_id} with priority {priority} for agent {self.agent_id}"
            )

            # Notify streaming subscribers
            await self._notify_streaming_subscribers(
                {
                    "event": "message_queued",
                    "message_id": queued_msg.message_id,
                    "priority": priority.value,
                    "queue_depth": self.stats.queue_depth,
                }
            )

        return queued_msg.message_id

    async def _process_message_immediate(self, queued_msg: QueuedMessage):
        """Process message immediately (streaming mode)"""
        queued_msg.status = QueuedMessageStatus.PROCESSING
        queued_msg.started_at = datetime.utcnow()

        try:
            with self._queue_lock:
                self._processing[queued_msg.message_id] = queued_msg
                self.stats.concurrent_processing = len(self._processing)

            # Process the message
            if self.message_processor:
                from ..core.a2aTypes import A2AMessage

                a2a_message = A2AMessage(**queued_msg.a2a_message)
                result = await self.message_processor(a2a_message, queued_msg.context_id)

                queued_msg.result = result
                queued_msg.status = QueuedMessageStatus.COMPLETED
            else:
                queued_msg.status = QueuedMessageStatus.FAILED
                queued_msg.error_message = "No message processor configured"

        except Exception as e:
            queued_msg.status = QueuedMessageStatus.FAILED
            queued_msg.error_message = str(e)
            logger.error(f"❌ Streaming processing failed for message {queued_msg.message_id}: {e}")

        finally:
            queued_msg.completed_at = datetime.utcnow()
            with self._queue_lock:
                if queued_msg.message_id in self._processing:
                    del self._processing[queued_msg.message_id]
                self._completed.append(queued_msg)
                self.stats.concurrent_processing = len(self._processing)
                self.stats.total_processed += 1

            # Update processing time stats
            if queued_msg.started_at and queued_msg.completed_at:
                processing_time = (queued_msg.completed_at - queued_msg.started_at).total_seconds()
                self.stats.avg_processing_time = (
                    self.stats.avg_processing_time * (self.stats.total_processed - 1)
                    + processing_time
                ) / self.stats.total_processed

    async def _queue_processor_loop(self):
        """Background queue processor loop"""
        while not self._shutdown_event.is_set():
            try:
                # Check if we can process more messages
                if len(self._processing) >= self.max_concurrent_processing:
                    await asyncio.sleep(0.1)
                    continue

                # Get next message from priority queue
                next_message = None
                with self._queue_lock:
                    while self._priority_queue:
                        _, _, message_id = heapq.heappop(self._priority_queue)
                        if message_id in self._messages:
                            next_message = self._messages.pop(message_id)
                            self.stats.queue_depth = len(self._messages)
                            break

                if not next_message:
                    await asyncio.sleep(0.1)
                    continue

                # Process the message
                task = asyncio.create_task(self._process_queued_message(next_message))
                self._processing_tasks[next_message.message_id] = task

            except Exception as e:
                logger.error(f"❌ Queue processor error for agent {self.agent_id}: {e}")
                await asyncio.sleep(1)

    async def _process_queued_message(self, queued_msg: QueuedMessage):
        """Process a message from the queue"""
        queued_msg.status = QueuedMessageStatus.PROCESSING
        queued_msg.started_at = datetime.utcnow()

        try:
            with self._queue_lock:
                self._processing[queued_msg.message_id] = queued_msg
                self.stats.concurrent_processing = len(self._processing)

            # Notify streaming subscribers
            await self._notify_streaming_subscribers(
                {
                    "event": "message_processing_started",
                    "message_id": queued_msg.message_id,
                    "priority": queued_msg.priority.value,
                }
            )

            # Process with timeout
            if self.message_processor:
                from ..core.a2aTypes import A2AMessage

                a2a_message = A2AMessage(**queued_msg.a2a_message)

                result = await asyncio.wait_for(
                    self.message_processor(a2a_message, queued_msg.context_id),
                    timeout=queued_msg.timeout_seconds,
                )

                queued_msg.result = result
                queued_msg.status = QueuedMessageStatus.COMPLETED

                # Notify streaming subscribers
                await self._notify_streaming_subscribers(
                    {
                        "event": "message_completed",
                        "message_id": queued_msg.message_id,
                        "processing_time": (
                            datetime.utcnow() - queued_msg.started_at
                        ).total_seconds(),
                    }
                )
            else:
                queued_msg.status = QueuedMessageStatus.FAILED
                queued_msg.error_message = "No message processor configured"

        except asyncio.TimeoutError:
            queued_msg.status = QueuedMessageStatus.TIMEOUT
            queued_msg.error_message = (
                f"Processing timeout after {queued_msg.timeout_seconds} seconds"
            )
            self.stats.total_timeout += 1
            logger.warning(f"⏰ Message {queued_msg.message_id} timeout for agent {self.agent_id}")

        except Exception as e:
            queued_msg.status = QueuedMessageStatus.FAILED
            queued_msg.error_message = str(e)
            self.stats.total_failed += 1
            logger.error(f"❌ Message processing failed for {queued_msg.message_id}: {e}")

            # Retry logic
            if queued_msg.retry_count < queued_msg.max_retries:
                queued_msg.retry_count += 1
                queued_msg.status = QueuedMessageStatus.QUEUED
                # Re-queue with delay
                await asyncio.sleep(2**queued_msg.retry_count)  # Exponential backoff
                with self._queue_lock:
                    self._messages[queued_msg.message_id] = queued_msg
                    priority_weight = self._priority_weights[queued_msg.priority]
                    timestamp = time.time()
                    heapq.heappush(
                        self._priority_queue, (priority_weight, timestamp, queued_msg.message_id)
                    )
                    self.stats.queue_depth = len(self._messages)
                logger.info(
                    f"🔄 Retrying message {queued_msg.message_id} (attempt {queued_msg.retry_count}/{queued_msg.max_retries})"
                )
                return

        finally:
            queued_msg.completed_at = datetime.utcnow()
            with self._queue_lock:
                if queued_msg.message_id in self._processing:
                    del self._processing[queued_msg.message_id]
                if queued_msg.message_id in self._processing_tasks:
                    del self._processing_tasks[queued_msg.message_id]
                self._completed.append(queued_msg)
                self.stats.concurrent_processing = len(self._processing)
                self.stats.total_processed += 1

            # Update processing time stats
            if queued_msg.started_at and queued_msg.completed_at:
                processing_time = (queued_msg.completed_at - queued_msg.started_at).total_seconds()
                self.stats.avg_processing_time = (
                    self.stats.avg_processing_time * (self.stats.total_processed - 1)
                    + processing_time
                ) / self.stats.total_processed

    async def subscribe_to_stream(self, subscriber_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to streaming updates about queue activity"""
        queue = asyncio.Queue(maxsize=100)
        self._streaming_subscribers[subscriber_id] = queue

        try:
            while True:
                event = await queue.get()
                yield event
        except asyncio.CancelledError:
            pass
        finally:
            if subscriber_id in self._streaming_subscribers:
                del self._streaming_subscribers[subscriber_id]

    async def _notify_streaming_subscribers(self, event: Dict[str, Any]):
        """Notify all streaming subscribers of an event"""
        if not self._streaming_subscribers:
            return

        for subscriber_id, queue in list(self._streaming_subscribers.items()):
            try:
                if queue.qsize() < queue.maxsize:
                    queue.put_nowait(event)
                else:
                    # Queue full, remove oldest item
                    try:
                        queue.get_nowait()
                        queue.put_nowait(event)
                    except asyncio.QueueEmpty:
                        pass
            except Exception as e:
                logger.warning(f"Failed to notify subscriber {subscriber_id}: {e}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics"""
        with self._queue_lock:
            queue_depth = len(self._messages)
            processing_count = len(self._processing)

            # Get priority breakdown
            priority_breakdown = {priority.value: 0 for priority in MessagePriority}
            for msg in self._messages.values():
                priority_breakdown[msg.priority.value] += 1

            # Get recent completed messages
            recent_completed = [
                {
                    "message_id": msg.message_id,
                    "status": msg.status.value,
                    "processing_time": (
                        (msg.completed_at - msg.started_at).total_seconds()
                        if msg.started_at and msg.completed_at
                        else None
                    ),
                    "priority": msg.priority.value,
                }
                for msg in list(self._completed)[-10:]  # Last 10 completed
            ]

        return {
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "queue_status": {
                "queue_depth": queue_depth,
                "processing_count": processing_count,
                "max_concurrent": self.max_concurrent_processing,
                "priority_breakdown": priority_breakdown,
            },
            "capabilities": {
                "streaming_enabled": self.enable_streaming,
                "batch_processing_enabled": self.enable_batch_processing,
                "auto_mode_threshold": self.auto_mode_threshold,
            },
            "statistics": self.stats.dict(),
            "recent_completed": recent_completed,
        }

    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific message"""
        with self._queue_lock:
            # Check processing
            if message_id in self._processing:
                return self._processing[message_id].dict()

            # Check queued
            if message_id in self._messages:
                return self._messages[message_id].dict()

            # Check completed
            for msg in self._completed:
                if msg.message_id == message_id:
                    return msg.dict()

        return None

    async def cancel_message(self, message_id: str) -> bool:
        """Cancel a queued or processing message"""
        with self._queue_lock:
            # Cancel queued message
            if message_id in self._messages:
                msg = self._messages.pop(message_id)
                msg.status = QueuedMessageStatus.CANCELLED
                msg.completed_at = datetime.utcnow()
                self._completed.append(msg)
                self.stats.queue_depth = len(self._messages)
                logger.info(f"🚫 Cancelled queued message {message_id}")
                return True

            # Cancel processing message
            if message_id in self._processing_tasks:
                task = self._processing_tasks[message_id]
                if not task.done():
                    task.cancel()
                    logger.info(f"🚫 Cancelled processing message {message_id}")
                    return True

        return False

    def clear_completed_history(self, older_than_hours: int = 24):
        """Clear completed message history older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        with self._queue_lock:
            # Filter completed messages
            filtered_completed = deque(maxlen=1000)
            for msg in self._completed:
                if msg.completed_at and msg.completed_at > cutoff_time:
                    filtered_completed.append(msg)

            self._completed = filtered_completed
            logger.info(f"🧹 Cleared completed message history older than {older_than_hours} hours")

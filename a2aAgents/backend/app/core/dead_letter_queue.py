"""
import time
Dead Letter Queue implementation for handling failed messages
"""

import json
import logging
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterMessage:
    """A message that failed processing"""
    message_id: str
    content: Dict[str, Any]
    error: str
    failure_count: int
    first_failure_time: datetime
    last_failure_time: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['first_failure_time'] = self.first_failure_time.isoformat()
        data['last_failure_time'] = self.last_failure_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeadLetterMessage':
        """Create from dictionary"""
        data['first_failure_time'] = datetime.fromisoformat(data['first_failure_time'])
        data['last_failure_time'] = datetime.fromisoformat(data['last_failure_time'])
        return cls(**data)


class DeadLetterQueue:
    """
    Persistent dead letter queue for failed messages
    """

    def __init__(self, storage_path: str = "./dlq", max_retries: int = 3):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.retry_handlers: Dict[str, Callable] = {}
        self.monitoring_callbacks: List[Callable] = []

    async def add_message(
        self,
        message_id: str,
        content: Dict[str, Any],
        error: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a failed message to the DLQ"""

        # Check if message already exists
        existing = await self.get_message(message_id)

        if existing:
            # Update existing message
            existing.failure_count += 1
            existing.last_failure_time = datetime.utcnow()
            existing.error = error
            message = existing
        else:
            # Create new message
            message = DeadLetterMessage(
                message_id=message_id,
                content=content,
                error=error,
                failure_count=1,
                first_failure_time=datetime.utcnow(),
                last_failure_time=datetime.utcnow(),
                source=source,
                metadata=metadata
            )

        # Save to disk
        await self._save_message(message)

        # Notify monitoring callbacks
        for callback in self.monitoring_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")

        logger.warning(
            f"Message {message_id} added to DLQ. "
            f"Failure count: {message.failure_count}, Error: {error}"
        )

    async def get_message(self, message_id: str) -> Optional[DeadLetterMessage]:
        """Retrieve a message from the DLQ"""
        file_path = self.storage_path / f"{message_id}.json"

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
                return DeadLetterMessage.from_dict(data)
        except Exception as e:
            logger.error(f"Error reading DLQ message {message_id}: {e}")
            return None

    async def _save_message(self, message: DeadLetterMessage):
        """Save message to disk"""
        file_path = self.storage_path / f"{message.message_id}.json"

        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(message.to_dict(), indent=2))

    async def list_messages(
        self,
        source: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[DeadLetterMessage]:
        """List messages in the DLQ"""
        messages = []

        for file_path in self.storage_path.glob("*.json"):
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                    message = DeadLetterMessage.from_dict(data)

                    # Apply filters
                    if source and message.source != source:
                        continue
                    if since and message.last_failure_time < since:
                        continue

                    messages.append(message)
            except Exception as e:
                logger.error(f"Error reading DLQ file {file_path}: {e}")

        return sorted(messages, key=lambda m: m.last_failure_time, reverse=True)

    async def retry_message(self, message_id: str) -> bool:
        """Attempt to retry a message"""
        message = await self.get_message(message_id)

        if not message:
            logger.error(f"Message {message_id} not found in DLQ")
            return False

        if message.failure_count >= self.max_retries:
            logger.error(
                f"Message {message_id} has exceeded max retries ({self.max_retries})"
            )
            return False

        # Get retry handler for source
        handler = self.retry_handlers.get(message.source)

        if not handler:
            logger.error(f"No retry handler registered for source: {message.source}")
            return False

        try:
            # Attempt retry
            await handler(message.content, message.metadata)

            # Remove from DLQ on success
            await self.remove_message(message_id)
            logger.info(f"Successfully retried message {message_id}")
            return True

        except Exception as e:
            # Update failure info
            await self.add_message(
                message_id=message_id,
                content=message.content,
                error=str(e),
                source=message.source,
                metadata=message.metadata
            )
            return False

    async def remove_message(self, message_id: str):
        """Remove a message from the DLQ"""
        file_path = self.storage_path / f"{message_id}.json"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Removed message {message_id} from DLQ")

    async def cleanup_old_messages(self, older_than_days: int = 30):
        """Remove messages older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        removed_count = 0

        messages = await self.list_messages()

        for message in messages:
            if message.last_failure_time < cutoff:
                await self.remove_message(message.message_id)
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} old messages from DLQ")
        return removed_count

    def register_retry_handler(self, source: str, handler: Callable):
        """Register a handler for retrying messages from a specific source"""
        self.retry_handlers[source] = handler

    def add_monitoring_callback(self, callback: Callable):
        """Add a callback to be notified when messages are added to DLQ"""
        self.monitoring_callbacks.append(callback)


# Global DLQ instance
dlq = DeadLetterQueue()

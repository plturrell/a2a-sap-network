"""
import time
Task persistence and recovery for agent tasks
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
from enum import Enum

from ..clients.sqliteClient import SQLiteClient, SQLiteConfig

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enum"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class PersistedTask:
    """Persisted task representation"""
    task_id: str
    agent_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        data['payload'] = json.dumps(self.payload)
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistedTask':
        """Create from dictionary"""
        data['status'] = TaskStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if isinstance(data.get('payload'), str):
            data['payload'] = json.loads(data['payload'])
        if isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)


class TaskPersistenceManager:
    """Manages task persistence and recovery"""

    def __init__(self, db_path: str = "./data/task_persistence.db"):
        self.db_path = db_path
        self.sqlite_client = SQLiteClient(
            SQLiteConfig(
                db_path=db_path,
                enable_encryption=False,  # Tasks don't need encryption
                pool_size=5
            )
        )
        self.task_handlers: Dict[str, Callable] = {}
        self.recovery_running = False
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize database schema for task persistence"""
        schema = """
        CREATE TABLE IF NOT EXISTS persisted_tasks (
            task_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error TEXT,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_agent_status ON persisted_tasks (agent_id, status);
        CREATE INDEX IF NOT EXISTS idx_created_at ON persisted_tasks (created_at);
        CREATE INDEX IF NOT EXISTS idx_status ON persisted_tasks (status);
        """

        with self.sqlite_client._get_connection() as conn:
            conn.executescript(schema)
            conn.commit()

    async def save_task(self, task: PersistedTask):
        """Save or update a task"""
        task.updated_at = datetime.utcnow()

        # Check if task exists
        existing = await self.get_task(task.task_id)

        if existing:
            # Update existing task
            await self.sqlite_client.async_update(
                table="persisted_tasks",
                data=task.to_dict(),
                filters={"task_id": task.task_id}
            )
        else:
            # Insert new task
            await self.sqlite_client.async_insert(
                table="persisted_tasks",
                data=task.to_dict()
            )

        logger.info(f"Saved task {task.task_id} with status {task.status.value}")

    async def get_task(self, task_id: str) -> Optional[PersistedTask]:
        """Get a task by ID"""
        result = await self.sqlite_client.async_select(
            table="persisted_tasks",
            filters={"task_id": task_id},
            limit=1
        )

        if result.success and result.data:
            return PersistedTask.from_dict(result.data[0])
        return None

    async def list_tasks(
        self,
        agent_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        since: Optional[datetime] = None
    ) -> List[PersistedTask]:
        """List tasks with optional filters"""
        filters = {}

        if agent_id:
            filters["agent_id"] = agent_id
        if status:
            filters["status"] = status.value

        result = await self.sqlite_client.async_select(
            table="persisted_tasks",
            filters=filters,
            order_by="-created_at",
            limit=1000
        )

        if not result.success:
            return []

        tasks = []
        for row in result.data:
            task = PersistedTask.from_dict(row)
            if since and task.created_at < since:
                continue
            tasks.append(task)

        return tasks

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None
    ):
        """Update task status"""
        task = await self.get_task(task_id)

        if not task:
            logger.error(f"Task {task_id} not found")
            return

        task.status = status
        task.updated_at = datetime.utcnow()

        if status == TaskStatus.IN_PROGRESS:
            task.started_at = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()

        if error:
            task.error = error

        await self.save_task(task)

    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def recover_tasks(self, agent_id: str):
        """Recover incomplete tasks for an agent"""
        if self.recovery_running:
            logger.warning("Task recovery already running")
            return

        self.recovery_running = True

        try:
            # Find incomplete tasks
            incomplete_statuses = [
                TaskStatus.PENDING,
                TaskStatus.IN_PROGRESS,
                TaskStatus.RETRYING
            ]

            tasks_to_recover = []
            for status in incomplete_statuses:
                tasks = await self.list_tasks(agent_id=agent_id, status=status)
                tasks_to_recover.extend(tasks)

            logger.info(f"Found {len(tasks_to_recover)} tasks to recover for agent {agent_id}")

            # Recover each task
            for task in tasks_to_recover:
                await self._recover_single_task(task)

        finally:
            self.recovery_running = False

    async def _recover_single_task(self, task: PersistedTask):
        """Recover a single task"""
        logger.info(f"Recovering task {task.task_id} of type {task.task_type}")

        # Check if handler exists
        handler = self.task_handlers.get(task.task_type)

        if not handler:
            logger.error(f"No handler registered for task type: {task.task_type}")
            await self.update_task_status(
                task.task_id,
                TaskStatus.FAILED,
                "No handler available for recovery"
            )
            return

        # Update retry count if it was in progress
        if task.status == TaskStatus.IN_PROGRESS:
            task.retry_count += 1

        # Check retry limit
        if task.retry_count >= task.max_retries:
            logger.error(f"Task {task.task_id} exceeded max retries")
            await self.update_task_status(
                task.task_id,
                TaskStatus.FAILED,
                "Max retries exceeded"
            )
            return

        try:
            # Update status to retrying
            await self.update_task_status(task.task_id, TaskStatus.RETRYING)

            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload, task.metadata)
            else:
                result = handler(task.payload, task.metadata)

            # Mark as completed
            await self.update_task_status(task.task_id, TaskStatus.COMPLETED)
            logger.info(f"Successfully recovered task {task.task_id}")

        except Exception as e:
            logger.error(f"Failed to recover task {task.task_id}: {e}")
            task.retry_count += 1

            if task.retry_count >= task.max_retries:
                await self.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    str(e)
                )
            else:
                await self.update_task_status(
                    task.task_id,
                    TaskStatus.PENDING,
                    str(e)
                )

    async def cleanup_old_tasks(self, older_than_days: int = 30):
        """Clean up old completed/failed tasks"""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)

        # Get old tasks
        old_tasks = await self.list_tasks()

        removed_count = 0
        for task in old_tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.completed_at and task.completed_at < cutoff:
                    # Delete task
                    result = self.sqlite_client.delete(
                        table="persisted_tasks",
                        filters={"task_id": task.task_id}
                    )
                    if result.success:
                        removed_count += 1

        logger.info(f"Cleaned up {removed_count} old tasks")
        return removed_count

    async def get_task_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get task statistics"""
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id

        result = await self.sqlite_client.async_select(
            table="persisted_tasks",
            filters=filters
        )

        if not result.success:
            return {}

        stats = {
            "total": len(result.data),
            "by_status": {},
            "by_type": {},
            "avg_completion_time": None
        }

        completion_times = []

        for row in result.data:
            status = row.get("status", "unknown")
            task_type = row.get("task_type", "unknown")

            # Count by status
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by type
            stats["by_type"][task_type] = stats["by_type"].get(task_type, 0) + 1

            # Calculate completion time
            if row.get("started_at") and row.get("completed_at"):
                started = datetime.fromisoformat(row["started_at"])
                completed = datetime.fromisoformat(row["completed_at"])
                completion_times.append((completed - started).total_seconds())

        # Calculate average completion time
        if completion_times:
            stats["avg_completion_time"] = sum(completion_times) / len(completion_times)

        return stats


# Global task persistence manager
task_manager = TaskPersistenceManager()

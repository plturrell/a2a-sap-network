#!/usr/bin/env python3
"""
Production Task Manager for A2A Agents
Provides persistent task management with retry logic, dead letter queue, and recovery.
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    DLQ = "dead_letter_queue"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PersistedTask:
    task_id: str
    agent_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    updated_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.metadata is None:
            self.metadata = {}


class DeadLetterQueue:
    """Dead Letter Queue for failed tasks."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the DLQ database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dlq_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT UNIQUE NOT NULL,
                        agent_id TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        original_error TEXT,
                        retry_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize DLQ database: {e}")
    
    async def add_message(self, task: PersistedTask, error: str = None):
        """Add a failed task to the dead letter queue."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO dlq_messages 
                    (task_id, agent_id, task_type, payload, original_error, retry_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    task.agent_id,
                    task.task_type,
                    json.dumps(task.payload),
                    error or task.error_message,
                    task.retry_count,
                    json.dumps(task.metadata or {})
                ))
                await conn.commit()
                logger.warning(f"Task {task.task_id} added to DLQ after {task.retry_count} retries")
        except Exception as e:
            logger.error(f"Failed to add task to DLQ: {e}")
    
    async def get_messages(self, limit: int = 100) -> List[Dict]:
        """Get messages from the dead letter queue."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute(
                    "SELECT * FROM dlq_messages ORDER BY created_at DESC LIMIT ?", 
                    (limit,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get DLQ messages: {e}")
            return []
    
    async def remove_message(self, task_id: str):
        """Remove a message from the DLQ."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("DELETE FROM dlq_messages WHERE task_id = ?", (task_id,))
                await conn.commit()
        except Exception as e:
            logger.error(f"Failed to remove DLQ message: {e}")


class TaskManager:
    """Production task manager with persistence and recovery."""
    
    def __init__(self, agent_id: str, db_path: Optional[str] = None):
        self.agent_id = agent_id
        self.db_path = db_path or f"/tmp/a2a_tasks_{agent_id}.db"
        self.task_handlers: Dict[str, Callable] = {}
        self.dlq = DeadLetterQueue(f"{self.db_path}.dlq")
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.init_db()
    
    def init_db(self):
        """Initialize the task database."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        next_retry_at TIMESTAMP,
                        error_message TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_retry ON tasks(next_retry_at)")
                
                conn.commit()
                logger.info(f"Task database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize task database: {e}")
            raise
    
    async def save_task(self, task: PersistedTask) -> Dict[str, str]:
        """Save a task to persistent storage."""
        try:
            task.updated_at = datetime.utcnow()
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO tasks (
                        task_id, agent_id, task_type, payload, status, priority,
                        created_at, updated_at, started_at, completed_at,
                        retry_count, max_retries, next_retry_at, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    task.agent_id,
                    task.task_type,
                    json.dumps(task.payload),
                    task.status.value,
                    task.priority.value,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.retry_count,
                    task.max_retries,
                    task.next_retry_at.isoformat() if task.next_retry_at else None,
                    task.error_message,
                    json.dumps(task.metadata or {})
                ))
                await conn.commit()
                
                logger.debug(f"Task {task.task_id} saved with status {task.status.value}")
                return {"task_id": task.task_id, "status": task.status.value}
        except Exception as e:
            logger.error(f"Failed to save task {task.task_id}: {e}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[PersistedTask]:
        """Get a task by ID."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_task(row, cursor.description)
                    return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                                error_message: Optional[str] = None,
                                metadata: Optional[Dict] = None):
        """Update task status."""
        try:
            task = await self.get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found for status update")
                return
            
            task.status = status
            task.updated_at = datetime.utcnow()
            
            if error_message:
                task.error_message = error_message
            
            if metadata:
                task.metadata = {**(task.metadata or {}), **metadata}
            
            if status == TaskStatus.PROCESSING:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = datetime.utcnow()
            
            await self.save_task(task)
            
            # Move to DLQ if failed after max retries
            if status == TaskStatus.FAILED and task.retry_count >= task.max_retries:
                task.status = TaskStatus.DLQ
                await self.save_task(task)
                await self.dlq.add_message(task, error_message)
            
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def process_task(self, task: PersistedTask) -> bool:
        """Process a single task."""
        handler = self.task_handlers.get(task.task_type)
        if not handler:
            logger.error(f"No handler registered for task type: {task.task_type}")
            await self.update_task_status(
                task.task_id, 
                TaskStatus.FAILED, 
                f"No handler for task type: {task.task_type}"
            )
            return False
        
        try:
            await self.update_task_status(task.task_id, TaskStatus.PROCESSING)
            
            # Execute the handler
            result = await handler(task.payload)
            
            await self.update_task_status(
                task.task_id, 
                TaskStatus.COMPLETED,
                metadata={"result": result}
            )
            
            logger.info(f"Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                # Schedule retry with exponential backoff
                retry_delay = min(2 ** task.retry_count * 60, 3600)  # Max 1 hour
                task.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                task.status = TaskStatus.RETRYING
                await self.save_task(task)
                logger.info(f"Task {task.task_id} scheduled for retry in {retry_delay} seconds")
            else:
                await self.update_task_status(task.task_id, TaskStatus.FAILED, str(e))
            
            return False
    
    async def recover_tasks(self, agent_id: str) -> List[PersistedTask]:
        """Recover pending and retrying tasks for an agent."""
        try:
            recovered_tasks = []
            
            async with aiosqlite.connect(self.db_path) as conn:
                # Get pending tasks
                async with conn.execute("""
                    SELECT * FROM tasks 
                    WHERE agent_id = ? AND status IN (?, ?) 
                    ORDER BY priority DESC, created_at ASC
                """, (agent_id, TaskStatus.PENDING.value, TaskStatus.PROCESSING.value)) as cursor:
                    async for row in cursor:
                        task = self._row_to_task(row, cursor.description)
                        recovered_tasks.append(task)
                
                # Get tasks ready for retry
                now = datetime.utcnow().isoformat()
                async with conn.execute("""
                    SELECT * FROM tasks 
                    WHERE agent_id = ? AND status = ? AND next_retry_at <= ?
                    ORDER BY priority DESC, next_retry_at ASC
                """, (agent_id, TaskStatus.RETRYING.value, now)) as cursor:
                    async for row in cursor:
                        task = self._row_to_task(row, cursor.description)
                        task.status = TaskStatus.PENDING  # Reset to pending for processing
                        recovered_tasks.append(task)
            
            logger.info(f"Recovered {len(recovered_tasks)} tasks for agent {agent_id}")
            return recovered_tasks
            
        except Exception as e:
            logger.error(f"Failed to recover tasks: {e}")
            return []
    
    async def create_task(self, task_type: str, payload: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         max_retries: int = 3) -> str:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        task = PersistedTask(
            task_id=task_id,
            agent_id=self.agent_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries
        )
        
        await self.save_task(task)
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_id
    
    def _row_to_task(self, row, description) -> PersistedTask:
        """Convert database row to PersistedTask object."""
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row))
        
        return PersistedTask(
            task_id=data['task_id'],
            agent_id=data['agent_id'],
            task_type=data['task_type'],
            payload=json.loads(data['payload']),
            status=TaskStatus(data['status']),
            priority=TaskPriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data['started_at'] else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
            retry_count=data['retry_count'],
            max_retries=data['max_retries'],
            next_retry_at=datetime.fromisoformat(data['next_retry_at']) if data['next_retry_at'] else None,
            error_message=data['error_message'],
            metadata=json.loads(data['metadata']) if data['metadata'] else {}
        )


# Global task manager instance
task_manager = TaskManager("default")


# Global DLQ instance  
dlq = task_manager.dlq
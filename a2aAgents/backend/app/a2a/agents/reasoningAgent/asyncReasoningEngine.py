"""
Async Reasoning Engine with Real Concurrency
Implements true concurrent processing for reasoning operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AsyncReasoningTask:
    task_id: str
    name: str
    coroutine: Coroutine
    priority: TaskPriority
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    completed_at: Optional[datetime] = None

class AsyncTaskScheduler:
    """Advanced async task scheduler with priority queues and dependency management"""

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.PriorityQueue(),
            TaskPriority.HIGH: asyncio.PriorityQueue(),
            TaskPriority.NORMAL: asyncio.PriorityQueue(),
            TaskPriority.LOW: asyncio.PriorityQueue()
        }

        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_dependencies = {}
        self.dependency_waiters = defaultdict(list)

        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.scheduler_running = False
        self.scheduler_task = None

        # Performance monitoring
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "throughput": 0.0
        }

        self.start_time = time.time()

    async def submit_task(self, task: AsyncReasoningTask) -> str:
        """Submit task to scheduler"""

        logger.debug(f"📋 Submitting task {task.task_id} with priority {task.priority.name}")

        # Store task dependencies
        if task.dependencies:
            self.task_dependencies[task.task_id] = task.dependencies.copy()

        # Add to appropriate priority queue
        priority_value = task.priority.value
        await self.task_queues[task.priority].put((priority_value, task.created_at.timestamp(), task))

        self.execution_stats["total_tasks"] += 1

        # Start scheduler if not running
        if not self.scheduler_running:
            await self.start_scheduler()

        return task.task_id

    async def start_scheduler(self):
        """Start the task scheduler"""

        if self.scheduler_running:
            return

        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("🚀 Async task scheduler started")

    async def stop_scheduler(self):
        """Stop the task scheduler"""

        self.scheduler_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("⏹️ Async task scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""

        while self.scheduler_running:
            try:
                # Get next task based on priority
                task = await self._get_next_ready_task()

                if task:
                    # Execute task with semaphore control
                    asyncio.create_task(self._execute_task_with_semaphore(task))
                else:
                    # No ready tasks, wait briefly
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1)

    async def _get_next_ready_task(self) -> Optional[AsyncReasoningTask]:
        """Get next task that's ready to run (dependencies satisfied)"""

        # Check priority queues in order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self.task_queues[priority]

            if not queue.empty():
                try:
                    # Non-blocking get to check if task is ready
                    _, _, task = queue.get_nowait()

                    # Check if dependencies are satisfied
                    if self._are_dependencies_satisfied(task.task_id):
                        return task
                    else:
                        # Put back in queue and try next
                        await queue.put((priority.value, task.created_at.timestamp(), task))

                except asyncio.QueueEmpty:
                    continue

        return None

    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""

        dependencies = self.task_dependencies.get(task_id, [])

        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False

            # Check if dependency completed successfully
            result = self.completed_tasks[dep_id]
            if result.status != TaskStatus.COMPLETED:
                return False

        return True

    async def _execute_task_with_semaphore(self, task: AsyncReasoningTask):
        """Execute task with semaphore control"""

        async with self.semaphore:
            await self._execute_task(task)

    async def _execute_task(self, task: AsyncReasoningTask):
        """Execute individual task"""

        start_time = time.time()
        task_result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)

        logger.debug(f"🔄 Executing task {task.task_id}: {task.name}")

        self.active_tasks[task.task_id] = task_result

        try:
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
            else:
                result = await task.coroutine

            # Task completed successfully
            execution_time = time.time() - start_time
            task_result.status = TaskStatus.COMPLETED
            task_result.result = result
            task_result.execution_time = execution_time
            task_result.completed_at = datetime.utcnow()

            self.execution_stats["completed_tasks"] += 1
            self._update_execution_stats(execution_time)

            logger.debug(f"✅ Task {task.task_id} completed in {execution_time:.2f}s")

        except asyncio.TimeoutError:
            task_result.status = TaskStatus.FAILED
            task_result.error = TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
            self.execution_stats["failed_tasks"] += 1
            logger.warning(f"⏰ Task {task.task_id} timed out")

        except asyncio.CancelledError:
            task_result.status = TaskStatus.CANCELLED
            logger.info(f"🚫 Task {task.task_id} cancelled")

        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error = e
            self.execution_stats["failed_tasks"] += 1
            logger.error(f"❌ Task {task.task_id} failed: {e}")

        finally:
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task_result
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            # Notify dependency waiters
            await self._notify_dependency_waiters(task.task_id)

    def _update_execution_stats(self, execution_time: float):
        """Update execution statistics"""

        completed = self.execution_stats["completed_tasks"]
        current_avg = self.execution_stats["avg_execution_time"]

        # Update rolling average
        self.execution_stats["avg_execution_time"] = (
            (current_avg * (completed - 1) + execution_time) / completed
        )

        # Calculate throughput (tasks per second)
        elapsed_time = time.time() - self.start_time
        self.execution_stats["throughput"] = completed / elapsed_time if elapsed_time > 0 else 0.0

    async def _notify_dependency_waiters(self, completed_task_id: str):
        """Notify tasks waiting on this dependency"""

        if completed_task_id in self.dependency_waiters:
            waiters = self.dependency_waiters[completed_task_id]
            for waiter_task_id in waiters:
                # Check if all dependencies are now satisfied
                if self._are_dependencies_satisfied(waiter_task_id):
                    logger.debug(f"🔗 Dependencies satisfied for task {waiter_task_id}")

            # Clear waiters
            del self.dependency_waiters[completed_task_id]

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for specific task to complete"""

        start_time = time.time()

        while True:
            # Check if task is completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            # Check if task exists
            if (task_id not in self.active_tasks and
                task_id not in self.completed_tasks and
                not self._task_in_queues(task_id)):
                raise ValueError(f"Task {task_id} not found")

            await asyncio.sleep(0.1)

    def _task_in_queues(self, task_id: str) -> bool:
        """Check if task is in any queue"""

        for queue in self.task_queues.values():
            # Note: This is a simplified check
            # In practice, you'd need to peek at queue contents
            if not queue.empty():
                return True
        return False

    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for all submitted tasks to complete"""

        start_time = time.time()

        while True:
            # Check if all tasks are completed
            total_submitted = self.execution_stats["total_tasks"]
            total_completed = len(self.completed_tasks)

            if total_completed >= total_submitted and not self.active_tasks:
                return self.completed_tasks.copy()

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks to complete")

            await asyncio.sleep(0.5)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""

        stats = self.execution_stats.copy()
        stats.update({
            "active_tasks": len(self.active_tasks),
            "pending_tasks": sum(q.qsize() for q in self.task_queues.values()),
            "scheduler_running": self.scheduler_running,
            "uptime": time.time() - self.start_time
        })

        return stats


class ConcurrentReasoningProcessor:
    """Concurrent processor for reasoning operations"""

    def __init__(self, max_workers: int = 4):
        self.scheduler = AsyncTaskScheduler(max_concurrent_tasks=max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers//2)

        # Operation queues for different types of work
        self.cpu_intensive_queue = asyncio.Queue(maxsize=100)
        self.io_intensive_queue = asyncio.Queue(maxsize=200)
        self.memory_intensive_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """Start the concurrent processor"""
        await self.scheduler.start_scheduler()
        logger.info("🔄 Concurrent reasoning processor started")

    async def stop(self):
        """Stop the concurrent processor"""
        await self.scheduler.stop_scheduler()
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("⏹️ Concurrent reasoning processor stopped")

    async def process_reasoning_pipeline(self, reasoning_stages: List[Dict[str, Any]]) -> List[Any]:
        """Process reasoning stages concurrently"""

        logger.info(f"🔀 Processing {len(reasoning_stages)} reasoning stages concurrently")

        # Create tasks for each stage
        tasks = []
        for i, stage in enumerate(reasoning_stages):
            stage_name = stage.get("name", f"stage_{i}")
            operation = stage.get("operation")
            params = stage.get("params", {})
            dependencies = stage.get("dependencies", [])

            if operation:
                task = AsyncReasoningTask(
                    task_id=f"reasoning_stage_{i}",
                    name=stage_name,
                    coroutine=self._execute_reasoning_operation(operation, params),
                    priority=TaskPriority.NORMAL,
                    dependencies=dependencies,
                    timeout=stage.get("timeout", 30)
                )

                task_id = await self.scheduler.submit_task(task)
                tasks.append(task_id)

        # Wait for all tasks to complete
        results = []
        for task_id in tasks:
            result = await self.scheduler.wait_for_task(task_id)
            if result.status == TaskStatus.COMPLETED:
                results.append(result.result)
            else:
                logger.error(f"Stage {task_id} failed: {result.error}")
                results.append(None)

        return results

    async def _execute_reasoning_operation(self, operation: Callable, params: Dict[str, Any]) -> Any:
        """Execute reasoning operation with appropriate executor"""

        operation_type = params.get("type", "cpu")

        if operation_type == "cpu_intensive":
            # Use process executor for CPU-intensive tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.process_executor, operation, params)

        elif operation_type == "io_intensive":
            # Use thread executor for I/O-intensive tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_executor, operation, params)

        else:
            # Regular async execution
            if asyncio.iscoroutinefunction(operation):
                return await operation(**params)
            else:
                return operation(**params)

    async def parallel_question_decomposition(self, questions: List[str],
                                            decomposer: Callable) -> List[Dict[str, Any]]:
        """Decompose multiple questions in parallel"""

        logger.info(f"🔀 Decomposing {len(questions)} questions in parallel")

        tasks = []
        for i, question in enumerate(questions):
            task = AsyncReasoningTask(
                task_id=f"decompose_{i}",
                name=f"decompose_question_{i}",
                coroutine=decomposer(question),
                priority=TaskPriority.NORMAL,
                timeout=20
            )

            task_id = await self.scheduler.submit_task(task)
            tasks.append(task_id)

        # Collect results
        decomposition_results = []
        for task_id in tasks:
            result = await self.scheduler.wait_for_task(task_id)
            if result.status == TaskStatus.COMPLETED:
                decomposition_results.append(result.result)
            else:
                decomposition_results.append({"error": str(result.error)})

        return decomposition_results

    async def concurrent_pattern_analysis(self, text_chunks: List[str],
                                        analyzer: Callable) -> List[Dict[str, Any]]:
        """Analyze patterns in text chunks concurrently"""

        logger.info(f"🔍 Analyzing patterns in {len(text_chunks)} chunks concurrently")

        # Create high-priority tasks for pattern analysis
        tasks = []
        for i, chunk in enumerate(text_chunks):
            task = AsyncReasoningTask(
                task_id=f"analyze_patterns_{i}",
                name=f"pattern_analysis_{i}",
                coroutine=analyzer(chunk),
                priority=TaskPriority.HIGH,
                timeout=15
            )

            task_id = await self.scheduler.submit_task(task)
            tasks.append(task_id)

        # Wait for all analyses to complete
        pattern_results = []
        for task_id in tasks:
            result = await self.scheduler.wait_for_task(task_id, timeout=20)
            if result.status == TaskStatus.COMPLETED:
                pattern_results.append(result.result)
            else:
                pattern_results.append({"patterns": [], "error": str(result.error)})

        return pattern_results

    async def parallel_agent_coordination(self, coordination_requests: List[Dict[str, Any]],
                                        coordinator: Callable) -> List[Dict[str, Any]]:
        """Coordinate multiple agent requests in parallel"""

        logger.info(f"🤝 Coordinating {len(coordination_requests)} agent requests in parallel")

        # Create tasks with different priorities based on request urgency
        tasks = []
        for i, request in enumerate(coordination_requests):
            urgency = request.get("urgency", "normal")
            priority = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL
            }.get(urgency, TaskPriority.NORMAL)

            task = AsyncReasoningTask(
                task_id=f"coordinate_{i}",
                name=f"agent_coordination_{i}",
                coroutine=coordinator(request),
                priority=priority,
                timeout=request.get("timeout", 60)
            )

            task_id = await self.scheduler.submit_task(task)
            tasks.append((task_id, request.get("callback")))

        # Collect results and trigger callbacks
        coordination_results = []
        for task_id, callback in tasks:
            result = await self.scheduler.wait_for_task(task_id)

            if result.status == TaskStatus.COMPLETED:
                coordination_results.append(result.result)

                # Trigger callback if provided
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result.result)
                        else:
                            callback(result.result)
                    except Exception as e:
                        logger.error(f"Callback failed for task {task_id}: {e}")
            else:
                coordination_results.append({"error": str(result.error)})

        return coordination_results

    async def stream_reasoning_results(self, reasoning_operations: List[Callable],
                                     result_callback: Callable) -> None:
        """Stream reasoning results as they become available"""

        logger.info(f"📡 Streaming results from {len(reasoning_operations)} operations")

        # Submit all operations
        tasks = []
        for i, operation in enumerate(reasoning_operations):
            task = AsyncReasoningTask(
                task_id=f"stream_op_{i}",
                name=f"streaming_operation_{i}",
                coroutine=operation(),
                priority=TaskPriority.NORMAL,
                timeout=30
            )

            task_id = await self.scheduler.submit_task(task)
            tasks.append(task_id)

        # Monitor and stream results as they complete
        completed_tasks = set()

        while len(completed_tasks) < len(tasks):
            for task_id in tasks:
                if task_id not in completed_tasks:
                    # Check if task completed
                    if task_id in self.scheduler.completed_tasks:
                        result = self.scheduler.completed_tasks[task_id]
                        completed_tasks.add(task_id)

                        # Stream result via callback
                        try:
                            if asyncio.iscoroutinefunction(result_callback):
                                await result_callback(task_id, result)
                            else:
                                result_callback(task_id, result)
                        except Exception as e:
                            logger.error(f"Stream callback failed for {task_id}: {e}")

            # Brief pause before next check
            await asyncio.sleep(0.1)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""

        scheduler_stats = self.scheduler.get_execution_stats()

        return {
            "scheduler": scheduler_stats,
            "thread_pool": {
                "max_workers": self.thread_executor._max_workers,
                "active_threads": len(self.thread_executor._threads)
            },
            "process_pool": {
                "max_workers": self.process_executor._max_workers,
                "active_processes": len(self.process_executor._processes) if hasattr(self.process_executor, '_processes') else 0
            },
            "queue_sizes": {
                "cpu_intensive": self.cpu_intensive_queue.qsize(),
                "io_intensive": self.io_intensive_queue.qsize(),
                "memory_intensive": self.memory_intensive_queue.qsize()
            }
        }


# Factory function for creating configured processor
def create_concurrent_reasoning_processor(max_workers: int = None) -> ConcurrentReasoningProcessor:
    """Create configured concurrent reasoning processor"""

    if max_workers is None:
        import os
        max_workers = min(os.cpu_count() or 4, 8)  # Reasonable default

    processor = ConcurrentReasoningProcessor(max_workers=max_workers)
    logger.info(f"Created concurrent reasoning processor with {max_workers} workers")

    return processor
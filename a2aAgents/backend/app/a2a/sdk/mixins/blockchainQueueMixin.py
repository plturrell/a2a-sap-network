"""
Blockchain Queue Mixin for A2A Agents
Provides standardized blockchain queue management capabilities to all agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from ...core.blockchainQueueManager import (
    BlockchainQueueManager, BlockchainTask, QueuePriority, TaskStatus,
    QueueType, get_blockchain_queue_manager, enqueue_a2a_task
)

logger = logging.getLogger(__name__)


class BlockchainQueueMixin:
    """
    Mixin to add blockchain queue management capabilities to A2A agents
    Automatically integrates with the A2A blockchain network for distributed task processing
    """

    def __init_blockchain_queue__(self, agent_id: str, blockchain_config: Optional[Dict[str, Any]] = None):
        """Initialize blockchain queue manager for this agent"""
        try:
            self.blockchain_queue = get_blockchain_queue_manager(agent_id, blockchain_config)
            self.agent_queue_name = f"agent_{agent_id}"
            self.queue_processing_active = False
            self.queue_processor_task = None

            # Queue event handlers
            self.queue_event_handlers: Dict[str, List[Callable]] = {
                "task_received": [],
                "task_completed": [],
                "task_failed": [],
                "queue_empty": [],
                "consensus_required": []
            }

            # Subscribe to agent-specific queue
            self.blockchain_queue.subscribe_to_queue(
                self.agent_queue_name,
                self._handle_queue_event,
                ["task_enqueued", "task_updated", "consensus_vote"]
            )

            logger.info(f"✅ Blockchain queue initialized for agent {agent_id}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize blockchain queue for {agent_id}: {e}")
            self.blockchain_queue = None

    async def start_queue_processing(self, max_concurrent: int = 5,
                                   poll_interval: float = 1.0):
        """
        Start automatic processing of blockchain queue tasks
        """
        if not self.blockchain_queue:
            logger.warning("Blockchain queue not available, skipping queue processing")
            return

        if self.queue_processing_active:
            logger.warning("Queue processing already active")
            return

        self.queue_processing_active = True
        self.queue_processor_task = asyncio.create_task(
            self._queue_processor_loop(max_concurrent, poll_interval)
        )

        logger.info(f"🚀 Started blockchain queue processing (max_concurrent: {max_concurrent})")

    async def stop_queue_processing(self):
        """Stop automatic queue processing"""
        if self.queue_processor_task:
            self.queue_processing_active = False
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass

            logger.info("🛑 Stopped blockchain queue processing")

    async def _queue_processor_loop(self, max_concurrent: int, poll_interval: float):
        """Main queue processing loop"""
        active_tasks = set()

        try:
            while self.queue_processing_active:
                try:
                    # Clean up completed tasks
                    completed_tasks = {task for task in active_tasks if task.done()}
                    active_tasks -= completed_tasks

                    # Process new tasks if we have capacity
                    while len(active_tasks) < max_concurrent:
                        # Dequeue next task
                        task = await self.blockchain_queue.dequeue_task(
                            self.agent_queue_name,
                            agent_filter=self.agent_id if hasattr(self, 'agent_id') else None
                        )

                        if not task:
                            break

                        # Process task asynchronously
                        processor_task = asyncio.create_task(
                            self._process_blockchain_task(task)
                        )
                        active_tasks.add(processor_task)

                    # Sync with blockchain periodically
                    if hasattr(self, '_last_blockchain_sync'):
                        if datetime.utcnow() - self._last_blockchain_sync > timedelta(minutes=5):
                            await self.blockchain_queue.sync_with_blockchain(self.agent_queue_name)
                            self._last_blockchain_sync = datetime.utcnow()
                    else:
                        self._last_blockchain_sync = datetime.utcnow()

                    # Wait before next poll
                    await asyncio.sleep(poll_interval)

                except Exception as e:
                    logger.error(f"Error in queue processor loop: {e}")
                    await asyncio.sleep(poll_interval * 2)  # Back off on error

        except asyncio.CancelledError:
            logger.info("Queue processor loop cancelled")
        finally:
            # Wait for active tasks to complete
            if active_tasks:
                logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _process_blockchain_task(self, task: BlockchainTask):
        """Process a single blockchain task"""
        try:
            # Update status to processing
            await self.blockchain_queue.update_task_status(
                task.task_id, TaskStatus.PROCESSING
            )

            logger.info(f"📝 Processing blockchain task: {task.task_id} ({task.skill_name})")

            # Execute the skill
            result = await self._execute_blockchain_skill(task)

            if result.get("success", False):
                # Mark as completed
                await self.blockchain_queue.update_task_status(
                    task.task_id, TaskStatus.COMPLETED, result=result
                )

                # Notify handlers
                await self._trigger_queue_event("task_completed", task, result)

                logger.info(f"✅ Blockchain task completed: {task.task_id}")
            else:
                # Mark as failed
                error_msg = result.get("error", "Unknown error")
                await self.blockchain_queue.update_task_status(
                    task.task_id, TaskStatus.FAILED, error_message=error_msg
                )

                # Handle retries
                if task.retry_count < task.max_retries:
                    await self._retry_blockchain_task(task)
                else:
                    await self._trigger_queue_event("task_failed", task, result)

                logger.error(f"❌ Blockchain task failed: {task.task_id} - {error_msg}")

        except Exception as e:
            logger.error(f"❌ Error processing blockchain task {task.task_id}: {e}")

            # Mark as failed
            await self.blockchain_queue.update_task_status(
                task.task_id, TaskStatus.FAILED, error_message=str(e)
            )

            # Trigger failure event
            await self._trigger_queue_event("task_failed", task, {"error": str(e)})

    async def _execute_blockchain_skill(self, task: BlockchainTask) -> Dict[str, Any]:
        """Execute skill requested in blockchain task"""
        try:
            skill_name = task.skill_name
            parameters = task.parameters

            # Check if agent has execute_skill method (from A2AAgentBase)
            if hasattr(self, 'execute_skill'):
                return await self.execute_skill(skill_name, parameters)

            # Check if skill exists as method
            if hasattr(self, skill_name):
                skill_method = getattr(self, skill_name)
                if callable(skill_method):
                    return await skill_method(parameters)

            # Check if skill exists in skills registry
            if hasattr(self, 'skills') and skill_name in self.skills:
                skill = self.skills[skill_name]
                return await skill(parameters)

            return {
                "success": False,
                "error": f"Skill '{skill_name}' not found in agent"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Skill execution failed: {str(e)}"
            }

    async def _retry_blockchain_task(self, task: BlockchainTask):
        """Retry a failed blockchain task"""
        task.retry_count += 1
        task.status = TaskStatus.RETRYING

        # Calculate retry delay (exponential backoff)
        retry_delay = min(300, 2 ** task.retry_count)  # Max 5 minutes
        task.scheduled_for = datetime.utcnow() + timedelta(seconds=retry_delay)

        # Re-enqueue for retry
        await self.blockchain_queue.enqueue_task(task, self.agent_queue_name)

        logger.info(f"🔄 Retrying blockchain task {task.task_id} in {retry_delay} seconds")

    async def send_a2a_blockchain_task(self, target_agent: str, skill_name: str,
                                     parameters: Dict[str, Any],
                                     priority: QueuePriority = QueuePriority.MEDIUM,
                                     timeout_seconds: int = 300) -> str:
        """
        Send A2A task via blockchain queue
        """
        if not self.blockchain_queue:
            raise RuntimeError("Blockchain queue not available")

        agent_id = self.agent_id if hasattr(self, 'agent_id') else "unknown_agent"

        task = BlockchainTask(
            sender_agent_id=agent_id,
            target_agent_id=target_agent,
            skill_name=skill_name,
            parameters=parameters,
            priority=priority,
            timeout_seconds=timeout_seconds,
            queue_type=QueueType.AGENT_DIRECT
        )

        task_id = await self.blockchain_queue.enqueue_task(task, f"agent_{target_agent}")

        logger.info(f"📤 Sent blockchain task to {target_agent}: {task_id} ({skill_name})")
        return task_id

    async def broadcast_a2a_blockchain_task(self, target_agents: List[str],
                                          skill_name: str, parameters: Dict[str, Any],
                                          priority: QueuePriority = QueuePriority.MEDIUM) -> List[str]:
        """
        Broadcast task to multiple agents via blockchain
        """
        if not self.blockchain_queue:
            raise RuntimeError("Blockchain queue not available")

        agent_id = self.agent_id if hasattr(self, 'agent_id') else "unknown_agent"

        task = BlockchainTask(
            sender_agent_id=agent_id,
            target_agents=target_agents,
            skill_name=skill_name,
            parameters=parameters,
            priority=priority,
            queue_type=QueueType.BROADCAST
        )

        task_ids = await self.blockchain_queue.broadcast_task(task, target_agents)

        logger.info(f"📢 Broadcast blockchain task to {len(target_agents)} agents: {skill_name}")
        return task_ids

    async def create_consensus_blockchain_task(self, participants: List[str],
                                             skill_name: str, parameters: Dict[str, Any],
                                             threshold: float = 0.67,
                                             priority: QueuePriority = QueuePriority.HIGH) -> str:
        """
        Create consensus-based blockchain task
        """
        if not self.blockchain_queue:
            raise RuntimeError("Blockchain queue not available")

        agent_id = self.agent_id if hasattr(self, 'agent_id') else "unknown_agent"

        task = BlockchainTask(
            sender_agent_id=agent_id,
            skill_name=skill_name,
            parameters=parameters,
            priority=priority,
            requires_consensus=True,
            consensus_threshold=threshold
        )

        task_id = await self.blockchain_queue.create_consensus_task(task, participants, threshold)

        logger.info(f"🤝 Created consensus blockchain task: {task_id} (participants: {len(participants)})")
        return task_id

    async def vote_on_consensus_task(self, task_id: str, vote: bool, reasoning: Optional[str] = None) -> bool:
        """
        Submit vote for consensus task
        """
        if not self.blockchain_queue:
            return False

        success = await self.blockchain_queue.vote_on_consensus_task(task_id, vote, reasoning)

        if success:
            logger.info(f"🗳️  Voted on consensus task {task_id}: {'approve' if vote else 'reject'}")

        return success

    def register_queue_event_handler(self, event_type: str, handler: Callable):
        """
        Register handler for queue events
        """
        if event_type in self.queue_event_handlers:
            self.queue_event_handlers[event_type].append(handler)
            logger.info(f"📋 Registered handler for queue event: {event_type}")

    async def _trigger_queue_event(self, event_type: str, task: BlockchainTask, data: Any = None):
        """Trigger queue event handlers"""
        if event_type in self.queue_event_handlers:
            for handler in self.queue_event_handlers[event_type]:
                try:
                    await handler(task, data)
                except Exception as e:
                    logger.error(f"Error in queue event handler: {e}")

    async def _handle_queue_event(self, queue_name: str, event_type: str, task: BlockchainTask):
        """Handle queue events from blockchain queue manager"""
        try:
            if event_type == "task_enqueued":
                await self._trigger_queue_event("task_received", task)
            elif event_type == "consensus_vote":
                await self._trigger_queue_event("consensus_required", task)

            # Log queue activity
            logger.debug(f"📫 Queue event: {event_type} for task {task.task_id} in {queue_name}")

        except Exception as e:
            logger.error(f"Error handling queue event: {e}")

    def get_blockchain_queue_metrics(self) -> Optional[Dict[str, Any]]:
        """Get blockchain queue performance metrics"""
        if not self.blockchain_queue:
            return None

        return self.blockchain_queue.get_all_metrics()

    async def sync_blockchain_queue(self) -> bool:
        """Manually sync queue with blockchain"""
        if not self.blockchain_queue:
            return False

        return await self.blockchain_queue.sync_with_blockchain(self.agent_queue_name)

    async def get_pending_blockchain_tasks(self) -> List[BlockchainTask]:
        """Get list of pending tasks in blockchain queue"""
        if not self.blockchain_queue or self.agent_queue_name not in self.blockchain_queue.local_queues:
            return []

        queue = self.blockchain_queue.local_queues[self.agent_queue_name]
        return [task for task in queue if task.status == TaskStatus.PENDING]

    async def cancel_blockchain_task(self, task_id: str) -> bool:
        """Cancel a pending blockchain task"""
        if not self.blockchain_queue:
            return False

        return await self.blockchain_queue.update_task_status(task_id, TaskStatus.CANCELLED)
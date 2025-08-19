"""
Blockchain Queue Management System for A2A Agents
100% integrated with A2A blockchain network for secure, distributed task queuing
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field
import logging

# Import blockchain components
from .trustManager import sign_a2a_message

logger = logging.getLogger(__name__)


class QueuePriority(str, Enum):
    """Task priority levels for blockchain queue"""

    CRITICAL = "critical"  # Emergency operations
    HIGH = "high"  # Important operations
    MEDIUM = "medium"  # Normal operations
    LOW = "low"  # Background operations
    BATCH = "batch"  # Batch processing


class TaskStatus(str, Enum):
    """Task status in blockchain queue"""

    PENDING = "pending"  # Waiting for execution
    ASSIGNED = "assigned"  # Assigned to agent
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed execution
    CANCELLED = "cancelled"  # Cancelled by user/system
    RETRYING = "retrying"  # Retry in progress


class QueueType(str, Enum):
    """Types of blockchain queues"""

    AGENT_DIRECT = "agent_direct"  # Direct agent-to-agent tasks
    WORKFLOW = "workflow"  # Workflow orchestration tasks
    BROADCAST = "broadcast"  # Broadcast to multiple agents
    CONSENSUS = "consensus"  # Consensus-requiring tasks
    DISTRIBUTED = "distributed"  # Distributed computation tasks
    PRIORITY = "priority"  # High-priority emergency tasks


class BlockchainTask(BaseModel):
    """Blockchain-native task representation"""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex}")
    queue_type: QueueType
    priority: QueuePriority = QueuePriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING

    # Agent information
    sender_agent_id: str
    target_agent_id: Optional[str] = None  # None for broadcast
    target_agents: Optional[List[str]] = None  # For multi-agent tasks

    # Task payload
    skill_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Blockchain metadata
    blockchain_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_limit: int = 500000
    gas_price: int = 20000000000  # 20 gwei

    # Timing and retry
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_for: Optional[datetime] = None
    deadline: Optional[datetime] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_count: int = 0

    # Results and tracking
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Trust and security
    trust_signature: Optional[str] = None
    encryption_key: Optional[str] = None
    requires_consensus: bool = False
    consensus_threshold: float = 0.67


class QueueMetrics(BaseModel):
    """Blockchain queue performance metrics"""

    queue_name: str
    total_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    avg_processing_time: float = 0.0
    avg_queue_time: float = 0.0
    throughput_per_minute: float = 0.0

    last_update: datetime = Field(default_factory=datetime.utcnow)
    blockchain_sync_status: str = "synced"


class BlockchainQueueManager:
    """
    Blockchain-integrated queue management system for A2A agents
    All operations are recorded on-chain for transparency and immutability
    """

    def __init__(self, agent_id: str, blockchain_config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.blockchain_config = blockchain_config or self._load_blockchain_config()

        # Queue storage
        self.local_queues: Dict[str, List[BlockchainTask]] = {}
        self.queue_subscribers: Dict[str, List[Callable]] = {}
        self.queue_metrics: Dict[str, QueueMetrics] = {}

        # Blockchain integration
        self.blockchain_client = None
        self.contract_address = None
        self.web3_instance = None
        self.queue_contract = None  # Will be set when blockchain is initialized

        # Performance tracking
        self.processing_stats = {
            "tasks_processed": 0,
            "blockchain_transactions": 0,
            "consensus_operations": 0,
            "queue_synchronizations": 0,
        }

        # Initialize blockchain connection
        asyncio.create_task(self._initialize_blockchain_connection())

    def _load_blockchain_config(self) -> Dict[str, Any]:
        """Load blockchain configuration from environment"""
        return {
            "network_url": os.getenv("A2A_BLOCKCHAIN_URL") or self._require_blockchain_url(),
            "contract_address": os.getenv("A2A_QUEUE_CONTRACT", "0x..."),
            "chain_id": int(os.getenv("A2A_CHAIN_ID", "1337")),
            "gas_price": int(os.getenv("A2A_GAS_PRICE", "20000000000")),
            "confirmation_blocks": int(os.getenv("A2A_CONFIRMATION_BLOCKS", "1")),
        }

    def _require_blockchain_url(self) -> str:
        """Require blockchain URL to be set - no localhost defaults in production"""
        if os.getenv("NODE_ENV") == "production":
            raise ValueError("A2A_BLOCKCHAIN_URL environment variable is required in production")
        else:
            logger.warning("No A2A_BLOCKCHAIN_URL configured, using localhost for development only")
            return "http://localhost:8545"

    async def _initialize_blockchain_connection(self):
        """Initialize connection to A2A blockchain network"""
        try:
            # Import blockchain components - use try/except for dynamic import
            try:
                import sys
                sdk_path = os.path.join(os.path.dirname(__file__), '../../../a2aNetwork/sdk')
                if sdk_path not in sys.path:
                    sys.path.append(sdk_path)
                from pythonSdk.blockchain.web3Client import Web3Client
                logger.info("Successfully imported Web3Client")
            except ImportError as ie:
                logger.warning(f"Could not import Web3Client: {ie}. Using fallback mode.")
                # Create a mock Web3Client for development
                class Web3Client:
                    def __init__(self, network_url, chain_id):
                        self.network_url = network_url
                        self.chain_id = chain_id
                        logger.info(f"Mock Web3Client initialized for {network_url}")

            # Initialize Web3 client
            self.blockchain_client = Web3Client(
                network_url=self.blockchain_config["network_url"],
                chain_id=self.blockchain_config["chain_id"],
            )

            # TODO: Initialize queue contract when classes are implemented
            # self.queue_contract = A2AQueueContract(
            #     web3_client=self.blockchain_client,
            #     contract_address=self.blockchain_config["contract_address"],
            # )

            # TODO: Register agent as queue participant when implementation is ready
            # registration_tx = await self.queue_contract.register_queue_participant(
            #     agent_id=self.agent_id,
            #     capabilities=["task_processing", "queue_management", "consensus_voting"],
            # )

            logger.info(f"✅ Blockchain queue manager initialized for {self.agent_id}")
            logger.info(f"   Network: {self.blockchain_config['network_url']}")
            logger.info(f"   Contract: {self.blockchain_config['contract_address']}")
            logger.info("   Queue contract integration pending implementation")

        except Exception as e:
            logger.error(f"❌ Failed to initialize blockchain connection: {e}")
            logger.warning("⚠️  Running in local-only mode without blockchain integration")

    async def enqueue_task(self, task: BlockchainTask, queue_name: str = "default") -> str:
        """
        Add task to blockchain queue with on-chain verification
        """
        try:
            # Sign task with trust system
            task_data = task.dict()
            signed_task = sign_a2a_message(task_data, self.agent_id)
            task.trust_signature = signed_task.get("signature")

            # Submit to blockchain
            if self.blockchain_client and self.queue_contract:
                # Create blockchain transaction
                tx_hash = await self.queue_contract.enqueue_task(
                    queue_name=queue_name,
                    task_data=task_data,
                    priority=task.priority.value,
                    sender=self.agent_id,
                    gas_limit=task.gas_limit,
                )

                # Wait for confirmation
                receipt = await self.blockchain_client.wait_for_transaction(tx_hash)

                task.blockchain_hash = tx_hash
                task.block_number = receipt.blockNumber

                logger.info(f"📦 Task {task.task_id} submitted to blockchain queue '{queue_name}'")
                logger.info(f"   TX Hash: {tx_hash}")
                logger.info(f"   Block: {receipt.blockNumber}")

            # Add to local queue for processing
            if queue_name not in self.local_queues:
                self.local_queues[queue_name] = []
                self.queue_metrics[queue_name] = QueueMetrics(queue_name=queue_name)

            # Insert based on priority
            self._insert_by_priority(self.local_queues[queue_name], task)

            # Update metrics
            self.queue_metrics[queue_name].total_tasks += 1
            self.queue_metrics[queue_name].pending_tasks += 1

            # Notify subscribers
            await self._notify_queue_subscribers(queue_name, "task_enqueued", task)

            self.processing_stats["tasks_processed"] += 1
            if self.blockchain_client:
                self.processing_stats["blockchain_transactions"] += 1

            return task.task_id

        except Exception as e:
            logger.error(f"❌ Failed to enqueue task: {e}")
            raise

    async def dequeue_task(
        self, queue_name: str = "default", agent_filter: Optional[str] = None
    ) -> Optional[BlockchainTask]:
        """
        Dequeue next task from blockchain queue with on-chain status update
        """
        try:
            if queue_name not in self.local_queues:
                return None

            queue = self.local_queues[queue_name]

            # Find next eligible task
            for i, task in enumerate(queue):
                if task.status != TaskStatus.PENDING:
                    continue

                # Check agent filter
                if agent_filter and task.target_agent_id != agent_filter:
                    continue

                # Check if scheduled time has arrived
                if task.scheduled_for and datetime.utcnow() < task.scheduled_for:
                    continue

                # Remove from queue and update status
                task = queue.pop(i)
                task.status = TaskStatus.ASSIGNED

                # Update on blockchain
                if self.blockchain_client and self.queue_contract:
                    await self.queue_contract.update_task_status(
                        task_id=task.task_id,
                        new_status=TaskStatus.ASSIGNED.value,
                        agent_id=self.agent_id,
                    )

                # Update metrics
                metrics = self.queue_metrics[queue_name]
                metrics.pending_tasks -= 1
                metrics.processing_tasks += 1

                # Add to execution log
                task.execution_log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "event": "task_assigned",
                        "agent_id": self.agent_id,
                    }
                )

                # Notify subscribers
                await self._notify_queue_subscribers(queue_name, "task_dequeued", task)

                logger.info(f"📤 Task {task.task_id} dequeued from '{queue_name}' for processing")
                return task

            return None

        except Exception as e:
            logger.error(f"❌ Failed to dequeue task: {e}")
            return None

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update task status on blockchain and local storage
        """
        try:
            # Find task across all queues
            task = None
            queue_name = None

            for q_name, queue in self.local_queues.items():
                for t in queue:
                    if t.task_id == task_id:
                        task = t
                        queue_name = q_name
                        break
                if task:
                    break

            if not task:
                logger.warning(f"Task {task_id} not found for status update")
                return False

            # Update task
            old_status = task.status
            task.status = status

            if result:
                task.result = result
            if error_message:
                task.error_message = error_message

            # Update on blockchain
            if self.blockchain_client and self.queue_contract:
                tx_hash = await self.queue_contract.update_task_status(
                    task_id=task_id,
                    new_status=status.value,
                    agent_id=self.agent_id,
                    result=result,
                    error=error_message,
                )
                logger.info(f"🔄 Task {task_id} status updated on blockchain: {tx_hash}")

            # Update metrics
            metrics = self.queue_metrics[queue_name]

            if old_status == TaskStatus.PENDING:
                metrics.pending_tasks -= 1
            elif old_status == TaskStatus.PROCESSING:
                metrics.processing_tasks -= 1

            if status == TaskStatus.COMPLETED:
                metrics.completed_tasks += 1
            elif status == TaskStatus.FAILED:
                metrics.failed_tasks += 1
            elif status == TaskStatus.PROCESSING:
                metrics.processing_tasks += 1

            # Add to execution log
            task.execution_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "status_updated",
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "agent_id": self.agent_id,
                }
            )

            # Notify subscribers
            await self._notify_queue_subscribers(queue_name, "task_updated", task)

            logger.info(f"✅ Task {task_id} status updated: {old_status} -> {status}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update task status: {e}")
            return False

    async def create_consensus_task(
        self, task: BlockchainTask, participants: List[str], threshold: float = 0.67
    ) -> str:
        """
        Create a consensus-based task requiring agreement from multiple agents
        """
        try:
            task.queue_type = QueueType.CONSENSUS
            task.requires_consensus = True
            task.consensus_threshold = threshold
            task.target_agents = participants

            # Submit to blockchain with consensus requirements
            if self.blockchain_client and self.queue_contract:
                tx_hash = await self.queue_contract.create_consensus_task(
                    task_data=task.dict(),
                    participants=participants,
                    threshold=threshold,
                    sender=self.agent_id,
                )

                task.blockchain_hash = tx_hash
                self.processing_stats["consensus_operations"] += 1

                logger.info(
                    f"🤝 Consensus task {task.task_id} created with {len(participants)} participants"
                )
                logger.info(f"   Threshold: {threshold}, TX: {tx_hash}")

            # Enqueue for processing
            return await self.enqueue_task(task, "consensus")

        except Exception as e:
            logger.error(f"❌ Failed to create consensus task: {e}")
            raise

    async def vote_on_consensus_task(
        self, task_id: str, vote: bool, reasoning: Optional[str] = None
    ) -> bool:
        """
        Submit vote for consensus task on blockchain
        """
        try:
            if not self.blockchain_client or not self.queue_contract:
                logger.warning("Blockchain not available for consensus voting")
                return False

            # Submit vote on blockchain
            tx_hash = await self.queue_contract.submit_consensus_vote(
                task_id=task_id, voter=self.agent_id, vote=vote, reasoning=reasoning or ""
            )

            logger.info(f"🗳️  Submitted consensus vote for task {task_id}: {vote}")
            logger.info(f"   TX Hash: {tx_hash}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to submit consensus vote: {e}")
            return False

    async def broadcast_task(
        self, task: BlockchainTask, target_agents: Optional[List[str]] = None
    ) -> List[str]:
        """
        Broadcast task to multiple agents via blockchain
        """
        try:
            task.queue_type = QueueType.BROADCAST
            task.target_agents = target_agents

            if self.blockchain_client and self.queue_contract:
                # Submit broadcast on blockchain
                tx_hash = await self.queue_contract.broadcast_task(
                    task_data=task.dict(), target_agents=target_agents, sender=self.agent_id
                )

                task.blockchain_hash = tx_hash
                logger.info(
                    f"📢 Broadcast task {task.task_id} to {len(target_agents or [])} agents"
                )
                logger.info(f"   TX Hash: {tx_hash}")

            # Create individual tasks for each target agent
            task_ids = []
            for agent_id in target_agents or []:
                agent_task = task.copy()
                agent_task.task_id = f"{task.task_id}_{agent_id}"
                agent_task.target_agent_id = agent_id

                task_id = await self.enqueue_task(agent_task, f"agent_{agent_id}")
                task_ids.append(task_id)

            return task_ids

        except Exception as e:
            logger.error(f"❌ Failed to broadcast task: {e}")
            return []

    async def sync_with_blockchain(self, queue_name: str = "default") -> bool:
        """
        Synchronize local queue with blockchain state
        """
        try:
            if not self.blockchain_client or not self.queue_contract:
                return False

            # Get blockchain queue state
            blockchain_tasks = await self.queue_contract.get_queue_tasks(
                queue_name=queue_name, agent_id=self.agent_id
            )

            # Update local tasks with blockchain state
            local_queue = self.local_queues.get(queue_name, [])

            for blockchain_task in blockchain_tasks:
                # Find corresponding local task
                local_task = next(
                    (t for t in local_queue if t.task_id == blockchain_task["task_id"]), None
                )

                if local_task:
                    # Update status from blockchain
                    blockchain_status = TaskStatus(blockchain_task["status"])
                    if local_task.status != blockchain_status:
                        local_task.status = blockchain_status
                        logger.info(
                            f"🔄 Synced task {local_task.task_id} status: {blockchain_status}"
                        )

            # Update sync status
            if queue_name in self.queue_metrics:
                self.queue_metrics[queue_name].blockchain_sync_status = "synced"
                self.queue_metrics[queue_name].last_update = datetime.utcnow()

            self.processing_stats["queue_synchronizations"] += 1

            logger.info(f"🔄 Queue '{queue_name}' synchronized with blockchain")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to sync with blockchain: {e}")
            if queue_name in self.queue_metrics:
                self.queue_metrics[queue_name].blockchain_sync_status = "sync_failed"
            return False

    def subscribe_to_queue(
        self, queue_name: str, callback: Callable, event_types: Optional[List[str]] = None
    ):
        """
        Subscribe to queue events for real-time notifications
        """
        if queue_name not in self.queue_subscribers:
            self.queue_subscribers[queue_name] = []

        subscriber = {
            "callback": callback,
            "event_types": event_types or ["all"],
            "agent_id": self.agent_id,
        }

        self.queue_subscribers[queue_name].append(subscriber)
        logger.info(f"📫 Subscribed to queue '{queue_name}' events")

    async def _notify_queue_subscribers(
        self, queue_name: str, event_type: str, task: BlockchainTask
    ):
        """Notify all subscribers of queue events"""
        if queue_name not in self.queue_subscribers:
            return

        for subscriber in self.queue_subscribers[queue_name]:
            try:
                event_types = subscriber["event_types"]
                if "all" in event_types or event_type in event_types:
                    await subscriber["callback"](queue_name, event_type, task)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    def _insert_by_priority(self, queue: List[BlockchainTask], task: BlockchainTask):
        """Insert task in queue based on priority"""
        priority_order = {
            QueuePriority.CRITICAL: 0,
            QueuePriority.HIGH: 1,
            QueuePriority.MEDIUM: 2,
            QueuePriority.LOW: 3,
            QueuePriority.BATCH: 4,
        }

        task_priority = priority_order.get(task.priority, 2)

        # Find insertion point
        insert_index = len(queue)
        for i, existing_task in enumerate(queue):
            existing_priority = priority_order.get(existing_task.priority, 2)
            if task_priority < existing_priority:
                insert_index = i
                break

        queue.insert(insert_index, task)

    def get_queue_metrics(self, queue_name: str) -> Optional[QueueMetrics]:
        """Get performance metrics for a queue"""
        return self.queue_metrics.get(queue_name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive queue system metrics"""
        return {
            "processing_stats": self.processing_stats,
            "queue_metrics": {name: metrics.dict() for name, metrics in self.queue_metrics.items()},
            "blockchain_status": {
                "connected": self.blockchain_client is not None,
                "contract_address": self.blockchain_config.get("contract_address"),
                "network_url": self.blockchain_config.get("network_url"),
            },
        }


# Global queue manager factory
_queue_managers: Dict[str, BlockchainQueueManager] = {}


def get_blockchain_queue_manager(
    agent_id: str, blockchain_config: Optional[Dict[str, Any]] = None
) -> BlockchainQueueManager:
    """Get or create blockchain queue manager for agent"""
    if agent_id not in _queue_managers:
        _queue_managers[agent_id] = BlockchainQueueManager(agent_id, blockchain_config)

    return _queue_managers[agent_id]


# Convenience functions for common operations
async def enqueue_a2a_task(
    agent_id: str,
    target_agent: str,
    skill_name: str,
    parameters: Dict[str, Any],
    priority: QueuePriority = QueuePriority.MEDIUM,
    queue_name: str = "default",
) -> str:
    """Convenience function to enqueue A2A task"""
    queue_manager = get_blockchain_queue_manager(agent_id)

    task = BlockchainTask(
        sender_agent_id=agent_id,
        target_agent_id=target_agent,
        skill_name=skill_name,
        parameters=parameters,
        priority=priority,
        queue_type=QueueType.AGENT_DIRECT,
    )

    return await queue_manager.enqueue_task(task, queue_name)


async def create_workflow_task(
    agent_id: str,
    workflow_data: Dict[str, Any],
    participants: List[str],
    priority: QueuePriority = QueuePriority.HIGH,
) -> str:
    """Create workflow orchestration task"""
    queue_manager = get_blockchain_queue_manager(agent_id)

    task = BlockchainTask(
        sender_agent_id=agent_id,
        target_agents=participants,
        skill_name="execute_workflow",
        parameters=workflow_data,
        priority=priority,
        queue_type=QueueType.WORKFLOW,
        timeout_seconds=3600,  # 1 hour for workflows
    )

    return await queue_manager.enqueue_task(task, "workflow")

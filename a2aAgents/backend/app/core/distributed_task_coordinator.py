"""
A2A-compliant distributed task coordination system
Provides Redis-based distributed task queue with failover and leader election
Integrates with A2A protocol, telemetry, and security systems
"""

import asyncio
import json
import logging
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from uuid import uuid4

# A2A SDK imports
from ..a2a.security.requestSigning import A2ARequestSigner

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simplified AI models")

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskState(str, Enum):
    """Distributed task states"""
    QUEUED = "queued"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class AILoadBalancingModel:
    """AI model for intelligent load balancing"""
    load_predictor: Optional[Any] = None
    performance_classifier: Optional[Any] = None
    scaler: Optional[Any] = None
    task_clusterer: Optional[Any] = None
    last_training: float = 0.0
    training_interval: float = 3600.0  # Retrain every hour
    prediction_accuracy: float = 0.0

    def __post_init__(self):
        if SKLEARN_AVAILABLE:
            self.load_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.performance_classifier = Ridge(alpha=0.1)
            self.scaler = StandardScaler()
            self.task_clusterer = KMeans(n_clusters=5, random_state=42)


@dataclass
class WorkerPerformanceMetrics:
    """AI-enhanced worker performance tracking"""
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    current_load: int = 0
    max_capacity: int = 10
    last_task_timestamp: float = 0.0
    performance_score: float = 1.0
    task_type_expertise: Dict[str, float] = None  # task_type -> success_rate
    recent_performance: deque = None  # Rolling window of recent task results
    predicted_capacity: float = 10.0  # AI-predicted optimal capacity
    workload_trend: float = 0.0  # Increasing/decreasing workload trend
    efficiency_score: float = 1.0  # Task completion efficiency
    collaboration_score: float = 1.0  # How well worker collaborates

    def __post_init__(self):
        if self.task_type_expertise is None:
            self.task_type_expertise = defaultdict(float)
        if self.recent_performance is None:
            self.recent_performance = deque(maxlen=50)  # Last 50 tasks


@dataclass
class DistributedTask:
    """Distributed task representation with AI enhancements"""
    task_id: str
    agent_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority
    state: TaskState
    created_at: datetime
    updated_at: datetime
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['state'] = self.state.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.claimed_at:
            data['claimed_at'] = self.claimed_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Create from dictionary"""
        data = data.copy()
        data['priority'] = TaskPriority(data['priority'])
        data['state'] = TaskState(data['state'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('claimed_at'):
            data['claimed_at'] = datetime.fromisoformat(data['claimed_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class DistributedLock:
    """Redis-based distributed lock"""

    def __init__(self, redis_client: redis.Redis, key: str, timeout: int = 30):
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.identifier = str(uuid4())
        self.acquired = False

    async def acquire(self) -> bool:
        """Acquire the distributed lock"""
        result = await self.redis.set(
            self.key,
            self.identifier,
            nx=True,
            ex=self.timeout
        )
        self.acquired = bool(result)
        return self.acquired

    async def release(self) -> bool:
        """Release the distributed lock"""
        if not self.acquired:
            return False

        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """

        result = await self.redis.eval(lua_script, 1, self.key, self.identifier)
        self.acquired = False
        return bool(result)

    async def extend(self, additional_time: int = 30) -> bool:
        """Extend the lock timeout"""
        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("EXPIRE", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        result = await self.redis.eval(
            lua_script, 1, self.key, self.identifier, additional_time
        )
        return bool(result)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class LeaderElection:
    """Redis-based leader election"""

    def __init__(self, redis_client: redis.Redis, election_key: str, node_id: str):
        self.redis = redis_client
        self.election_key = f"leader:{election_key}"
        self.node_id = node_id
        self.heartbeat_interval = 10
        self.election_timeout = 30
        self.is_leader = False
        self.heartbeat_task = None

    async def start_election(self):
        """Start the leader election process"""
        while True:
            try:
                # Try to become leader
                result = await self.redis.set(
                    self.election_key,
                    self.node_id,
                    nx=True,
                    ex=self.election_timeout
                )

                if result:
                    # Became leader
                    self.is_leader = True
                    logger.info(f"Node {self.node_id} became leader")
                    self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    break
                else:
                    # Check if current leader is alive
                    current_leader = await self.redis.get(self.election_key)
                    if not current_leader:
                        continue  # Try again

                    logger.info(f"Node {self.node_id} following leader: {current_leader.decode()}")
                    await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Leader election error: {e}")
                await asyncio.sleep(5)

    async def _heartbeat_loop(self):
        """Maintain leadership with heartbeats"""
        while self.is_leader:
            try:
                # Extend leadership
                lua_script = """
                if redis.call("GET", KEYS[1]) == ARGV[1] then
                    return redis.call("EXPIRE", KEYS[1], ARGV[2])
                else
                    return 0
                end
                """

                result = await self.redis.eval(
                    lua_script, 1, self.election_key, self.node_id, self.election_timeout
                )

                if not result:
                    # Lost leadership
                    self.is_leader = False
                    logger.warning(f"Node {self.node_id} lost leadership")
                    break

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.is_leader = False
                break

    async def stop(self):
        """Stop leader election"""
        self.is_leader = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Release leadership if we have it
        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """

        await self.redis.eval(lua_script, 1, self.election_key, self.node_id)


class DistributedTaskCoordinator:
    """Distributed task coordination system"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        node_id: str = None,
        task_namespace: str = "a2a_tasks"
    ):
        self.redis_url = redis_url
        self.node_id = node_id or f"node_{uuid4().hex[:8]}"
        self.namespace = task_namespace
        self.redis = None
        self.task_handlers: Dict[str, Callable] = {}
        self.processing_tasks: Set[str] = set()
        self.leader_election = None
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False

        # AI Load Balancing Components
        self.ai_model = AILoadBalancingModel()
        self.worker_metrics: Dict[str, WorkerPerformanceMetrics] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.system_load_history: deque = deque(maxlen=1000)  # System load over time
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # worker_id -> (predicted_load, timestamp)

        # Redis keys
        self.task_queue_key = f"{self.namespace}:queue"
        self.task_data_key = f"{self.namespace}:data"
        self.task_claims_key = f"{self.namespace}:claims"
        self.node_heartbeat_key = f"{self.namespace}:nodes"
        self.metrics_key = f"{self.namespace}:metrics"
        self.ai_model_key = f"{self.namespace}:ai_model"

    async def initialize(self):
        """Initialize the distributed task coordinator"""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

        # Start leader election
        self.leader_election = LeaderElection(
            self.redis,
            f"{self.namespace}_coordinator",
            self.node_id
        )

        # Start background tasks
        asyncio.create_task(self.leader_election.start_election())
        asyncio.create_task(self._node_heartbeat_loop())

        logger.info(f"Distributed task coordinator initialized for node {self.node_id}")

    async def start_workers(self, num_workers: int = 3):
        """Start AI-optimized worker processes"""
        self.running = True

        # Initialize AI components
        self.worker_metrics = {}
        self.task_completion_history = deque(maxlen=1000)
        self.load_prediction_model = self._initialize_load_predictor()

        for i in range(num_workers):
            worker_id = f"worker_{i}"
            self.worker_metrics[worker_id] = WorkerPerformanceMetrics(worker_id=worker_id)
            task = asyncio.create_task(self._ai_enhanced_worker_loop(worker_id))
            self.worker_tasks.append(task)

        # Start AI monitoring and optimization tasks
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._intelligent_load_balancing_loop())

        logger.info(f"Started {num_workers} AI-enhanced workers on node {self.node_id}")

    async def stop(self):
        """Stop the coordinator"""
        self.running = False

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Stop leader election
        if self.leader_election:
            await self.leader_election.stop()

        # Close Redis connection
        if self.redis:
            await self.redis.close()

        logger.info(f"Stopped distributed task coordinator for node {self.node_id}")

    def register_handler(self, task_type: str, handler: Callable):
        """Register a task handler"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def submit_task(
        self,
        agent_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: int = 300,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Submit a task to the distributed queue"""
        task_id = str(uuid4())

        task = DistributedTask(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            state=TaskState.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )

        # Store task data
        await self.redis.hset(
            self.task_data_key,
            task_id,
            json.dumps(task.to_dict())
        )

        # Add to priority queue with AI-enhanced scoring
        priority_score = await self._get_ai_enhanced_priority_score(task, priority)
        await self.redis.zadd(
            self.task_queue_key,
            {task_id: priority_score}
        )

        logger.info(f"Submitted task {task_id} of type {task_type} for agent {agent_id}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get task status"""
        task_data = await self.redis.hget(self.task_data_key, task_id)
        if not task_data:
            return None

        return DistributedTask.from_dict(json.loads(task_data))

    async def _worker_loop(self, worker_id: str):
        """Worker loop to process tasks"""
        logger.info(f"Worker {worker_id} started on node {self.node_id}")

        while self.running:
            try:
                # Claim a task
                task = await self._claim_task()
                if not task:
                    await asyncio.sleep(1)
                    continue

                # Process the task
                await self._process_task(task, worker_id)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)

        logger.info(f"Worker {worker_id} stopped")

    async def _claim_task(self) -> Optional[DistributedTask]:
        """Claim the next available task"""
        # Get highest priority task
        task_ids = await self.redis.zrevrange(self.task_queue_key, 0, 0)
        if not task_ids:
            return None

        task_id = task_ids[0]

        # Try to claim the task atomically
        claim_key = f"{self.task_claims_key}:{task_id}"
        claimed = await self.redis.set(
            claim_key,
            self.node_id,
            nx=True,
            ex=300  # 5 minute claim timeout
        )

        if not claimed:
            return None

        # Remove from queue
        await self.redis.zrem(self.task_queue_key, task_id)

        # Get task data
        task_data = await self.redis.hget(self.task_data_key, task_id)
        if not task_data:
            await self.redis.delete(claim_key)
            return None

        task = DistributedTask.from_dict(json.loads(task_data))
        task.state = TaskState.CLAIMED
        task.claimed_by = self.node_id
        task.claimed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()

        # Update task data
        await self.redis.hset(
            self.task_data_key,
            task_id,
            json.dumps(task.to_dict())
        )

        return task

    async def _process_task(self, task: DistributedTask, worker_id: str):
        """Process a claimed task"""
        task_id = task.task_id
        claim_key = f"{self.task_claims_key}:{task_id}"

        try:
            self.processing_tasks.add(task_id)

            # Update task state
            task.state = TaskState.PROCESSING
            task.started_at = datetime.utcnow()
            task.updated_at = datetime.utcnow()

            await self.redis.hset(
                self.task_data_key,
                task_id,
                json.dumps(task.to_dict())
            )

            # Get handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")

            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(task.payload, task.metadata),
                    timeout=task.timeout_seconds
                )

                # Mark as completed
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.utcnow()
                task.updated_at = datetime.utcnow()
                task.metadata = task.metadata or {}
                task.metadata['result'] = result

                logger.info(f"Task {task_id} completed successfully")

            except asyncio.TimeoutError:
                raise Exception(f"Task timed out after {task.timeout_seconds} seconds")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.state = TaskState.RETRYING
                task.updated_at = datetime.utcnow()
                task.error = str(e)

                # Re-queue with delay
                retry_delay = min(60 * (2 ** task.retry_count), 3600)  # Exponential backoff
                retry_score = time.time() + retry_delay
                await self.redis.zadd(self.task_queue_key, {task_id: retry_score})

                logger.info(f"Task {task_id} scheduled for retry {task.retry_count}/{task.max_retries}")
            else:
                # Move to dead letter
                task.state = TaskState.DEAD_LETTER
                task.updated_at = datetime.utcnow()
                task.error = str(e)

                logger.error(f"Task {task_id} moved to dead letter after {task.max_retries} retries")

        finally:
            # Update final task state
            await self.redis.hset(
                self.task_data_key,
                task_id,
                json.dumps(task.to_dict())
            )

            # Release claim
            await self.redis.delete(claim_key)
            self.processing_tasks.discard(task_id)

    async def _node_heartbeat_loop(self):
        """Send periodic heartbeats to indicate node health"""
        while True:
            try:
                heartbeat_data = {
                    'node_id': self.node_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'processing_tasks': len(self.processing_tasks),
                    'is_leader': self.leader_election.is_leader if self.leader_election else False
                }

                await self.redis.hset(
                    self.node_heartbeat_key,
                    self.node_id,
                    json.dumps(heartbeat_data)
                )

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_stale_tasks(self):
        """Cleanup stale tasks (leader responsibility)"""
        if not self.leader_election.is_leader:
            return

        try:
            # Find stale claims (older than 10 minutes)
            claim_pattern = f"{self.task_claims_key}:*"

            async for key in self.redis.scan_iter(match=claim_pattern):
                ttl = await self.redis.ttl(key)
                if ttl == -1:  # No expiration set, it's stale
                    task_id = key.split(':')[-1]
                    await self.redis.delete(key)

                    # Re-queue the task
                    task_data = await self.redis.hget(self.task_data_key, task_id)
                    if task_data:
                        task = DistributedTask.from_dict(json.loads(task_data))
                        if task.state in [TaskState.CLAIMED, TaskState.PROCESSING]:
                            task.state = TaskState.QUEUED
                            task.claimed_by = None
                            task.claimed_at = None
                            task.updated_at = datetime.utcnow()

                            # Re-add to queue with AI-enhanced priority
                            priority_score = await self._get_ai_priority_score(task)
                            await self.redis.zadd(self.task_queue_key, {task_id: priority_score})

                            # Update task data
                            await self.redis.hset(
                                self.task_data_key,
                                task_id,
                                json.dumps(task.to_dict())
                            )

                            logger.info(f"Re-queued stale task {task_id}")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def _get_priority_score(self, priority: TaskPriority) -> float:
        """Get priority score for queue ordering"""
        scores = {
            TaskPriority.LOW: 1.0,
            TaskPriority.NORMAL: 2.0,
            TaskPriority.HIGH: 3.0,
            TaskPriority.CRITICAL: 4.0
        }
        return scores.get(priority, 2.0) * 1000000 + time.time()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        stats = {
            'node_id': self.node_id,
            'is_leader': self.leader_election.is_leader if self.leader_election else False,
            'processing_tasks': len(self.processing_tasks),
            'registered_handlers': len(self.task_handlers),
            'queue_size': await self.redis.zcard(self.task_queue_key),
            'total_tasks': await self.redis.hlen(self.task_data_key)
        }

        # Get task state counts
        all_tasks = await self.redis.hgetall(self.task_data_key)
        state_counts = {}
        for task_data in all_tasks.values():
            task = DistributedTask.from_dict(json.loads(task_data))
            state_counts[task.state.value] = state_counts.get(task.state.value, 0) + 1

        stats['task_states'] = state_counts
        return stats


# Global coordinator instance
coordinator = DistributedTaskCoordinator()


async def initialize_distributed_coordination(
    redis_url: str = "redis://localhost:6379",
    node_id: str = None
):
    """Initialize the global distributed task coordinator"""
    global coordinator
    coordinator = DistributedTaskCoordinator(redis_url, node_id)
    await coordinator.initialize()
    await coordinator.start_workers()
    return coordinator


    def _initialize_load_predictor(self) -> Dict[str, Any]:
        """Initialize AI load prediction model"""
        if SKLEARN_AVAILABLE:
            return {
                'model': LinearRegression(),
                'scaler': StandardScaler(),
                'trained': False,
                'feature_history': deque(maxlen=200),
                'target_history': deque(maxlen=200)
            }
        else:
            return {
                'model': None,
                'simple_predictor': {'avg_load': 0.0, 'trend': 0.0},
                'history': deque(maxlen=50)
            }

    async def _ai_enhanced_worker_loop(self, worker_id: str):
        """AI-enhanced worker loop with intelligent task selection"""
        worker_metrics = self.worker_metrics[worker_id]

        while self.running:
            try:
                # Get optimal task for this worker using AI
                task = await self._get_optimal_task_for_worker(worker_id)

                if not task:
                    await asyncio.sleep(1)
                    continue

                # Update worker load
                worker_metrics.current_load += 1
                start_time = time.time()

                # Process task
                success = await self._process_task_with_monitoring(task, worker_id)

                # Update performance metrics
                processing_time = time.time() - start_time
                await self._update_worker_performance(worker_id, task, success, processing_time)

                # Update global statistics
                self.task_completion_history.append({
                    'worker_id': worker_id,
                    'task_type': task.task_type,
                    'success': success,
                    'processing_time': processing_time,
                    'timestamp': time.time()
                })

                worker_metrics.current_load -= 1

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                worker_metrics.current_load = max(0, worker_metrics.current_load - 1)
                await asyncio.sleep(5)

    async def _get_optimal_task_for_worker(self, worker_id: str) -> Optional[DistributedTask]:
        """Use AI to select the best task for a specific worker"""
        worker_metrics = self.worker_metrics[worker_id]

        # Skip if worker is at capacity
        if worker_metrics.current_load >= worker_metrics.max_capacity:
            return None

        # Get available tasks with intelligent scoring
        tasks = await self._get_available_tasks_with_ai_scoring(worker_id)

        if not tasks:
            return None

        # Select best task based on worker expertise and current conditions
        best_task = max(tasks, key=self._get_task_ai_score)

        # Claim the task
        task_id = best_task['task_id']
        if await self._claim_task(task_id, worker_id):
            return await self._load_task(task_id)

        return None

    async def _get_available_tasks_with_ai_scoring(self, worker_id: str) -> List[Dict[str, Any]]:
        """Get tasks with AI-enhanced scoring for worker affinity"""
        worker_metrics = self.worker_metrics[worker_id]

        # Get top tasks from priority queue
        task_ids = await self.redis.zrevrange(self.task_queue_key, 0, 20)

        scored_tasks = []
        for task_id in task_ids:
            task_data = await self.redis.hget(self.task_data_key, task_id)
            if not task_data:
                continue

            task_dict = json.loads(task_data)
            task_type = task_dict['task_type']

            # Calculate AI-enhanced affinity score
            base_priority = task_dict.get('priority_score', 0)
            worker_expertise = worker_metrics.task_type_expertise.get(task_type, 0.5)

            # Factor in worker performance and current load
            performance_factor = worker_metrics.performance_score
            load_factor = 1.0 - (worker_metrics.current_load / worker_metrics.max_capacity)

            # Recent success rate for this task type
            recent_success = self._calculate_recent_success_rate(worker_id, task_type)

            # Combined AI score
            ai_score = (base_priority * 0.4 +
                       worker_expertise * 100 * 0.3 +
                       performance_factor * 50 * 0.2 +
                       recent_success * 50 * 0.1) * load_factor

            scored_tasks.append({
                'task_id': task_id,
                'task_type': task_type,
                'ai_score': ai_score,
                'base_priority': base_priority,
                'worker_expertise': worker_expertise
            })

        return scored_tasks

    async def _update_worker_performance(self, worker_id: str, task: DistributedTask,
                                       success: bool, processing_time: float):
        """Update worker performance metrics with AI learning"""
        worker_metrics = self.worker_metrics[worker_id]

        # Update basic counters
        worker_metrics.total_tasks += 1
        if success:
            worker_metrics.successful_tasks += 1
        else:
            worker_metrics.failed_tasks += 1

        # Update average processing time
        if worker_metrics.total_tasks == 1:
            worker_metrics.avg_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            worker_metrics.avg_processing_time = (alpha * processing_time +
                                                 (1 - alpha) * worker_metrics.avg_processing_time)

        # Update task type expertise
        task_type = task.task_type
        current_expertise = worker_metrics.task_type_expertise[task_type]

        if success:
            # Increase expertise (with diminishing returns)
            new_expertise = min(1.0, current_expertise + (1.0 - current_expertise) * 0.1)
        else:
            # Decrease expertise slightly
            new_expertise = max(0.1, current_expertise * 0.95)

        worker_metrics.task_type_expertise[task_type] = new_expertise

        # Update recent performance window
        worker_metrics.recent_performance.append({
            'success': success,
            'processing_time': processing_time,
            'task_type': task_type,
            'timestamp': time.time()
        })

        # Calculate overall performance score
        if len(worker_metrics.recent_performance) >= 10:
            recent_successes = sum(1 for p in worker_metrics.recent_performance if p['success'])
            success_rate = recent_successes / len(worker_metrics.recent_performance)

            # Factor in processing time efficiency
            avg_time = sum(p['processing_time'] for p in worker_metrics.recent_performance) / len(worker_metrics.recent_performance)
            time_efficiency = max(0.1, min(2.0, 30.0 / avg_time))  # Normalize around 30 seconds

            worker_metrics.performance_score = (success_rate * 0.7 + time_efficiency * 0.3)

        worker_metrics.last_task_timestamp = time.time()

    def _calculate_recent_success_rate(self, worker_id: str, task_type: str) -> float:
        """Calculate recent success rate for worker + task type combination"""
        worker_metrics = self.worker_metrics[worker_id]

        recent_tasks_of_type = [p for p in worker_metrics.recent_performance
                               if p['task_type'] == task_type and
                               time.time() - p['timestamp'] < 3600]  # Last hour

        if len(recent_tasks_of_type) < 3:
            return worker_metrics.task_type_expertise.get(task_type, 0.5)

        successes = sum(1 for t in recent_tasks_of_type if t['success'])
        return successes / len(recent_tasks_of_type)

    async def _performance_monitoring_loop(self):
        """Monitor and optimize worker performance"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Analyze worker performance
                await self._analyze_worker_performance()

                # Update load prediction model
                await self._update_load_prediction_model()

                # Optimize worker allocation if needed
                await self._optimize_worker_allocation()

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _intelligent_load_balancing_loop(self):
        """Intelligent load balancing based on AI predictions"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Predict upcoming load
                predicted_load = await self._predict_system_load()

                # Adjust worker capacities based on predictions
                await self._adjust_worker_capacities(predicted_load)

                # Rebalance tasks if needed
                await self._rebalance_task_assignments()

                logger.debug(f"Load balancing: predicted_load={predicted_load:.2f}")

            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(120)

    async def _get_ai_enhanced_priority_score(self, task: DistributedTask, priority: TaskPriority) -> float:
        """Calculate AI-enhanced priority score considering system state"""
        base_score = self._get_priority_score(priority)

        # Factor in task type demand/supply ratio
        queue_size = await self.redis.zcard(self.task_queue_key)

        # Count workers capable of handling this task type
        capable_workers = sum(1 for m in self.worker_metrics.values()
                            if m.task_type_expertise.get(task.task_type, 0) > 0.3)

        if capable_workers == 0:
            capable_workers = len(self.worker_metrics)  # All workers can try

        # Adjust score based on supply/demand
        demand_factor = min(2.0, queue_size / max(1, capable_workers * 5))

        # Consider urgency based on task age and system load
        current_load = sum(m.current_load for m in self.worker_metrics.values())
        total_capacity = sum(m.max_capacity for m in self.worker_metrics.values())
        system_utilization = current_load / max(1, total_capacity)

        urgency_multiplier = 1.0
        if system_utilization > 0.8:  # High system load
            urgency_multiplier = 1.5 if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL] else 0.8

        enhanced_score = base_score * demand_factor * urgency_multiplier

        return enhanced_score

    async def _select_optimal_worker_ai(self, task: DistributedTask) -> Optional[str]:
        """AI-powered worker selection with predictive load balancing"""
        if not self.worker_metrics:
            return None

        current_time = time.time()

        # Update AI model if needed
        if current_time - self.ai_model.last_training > self.ai_model.training_interval:
            await self._train_ai_model()

        # Get predictions for each worker
        worker_scores = {}
        for worker_id, metrics in self.worker_metrics.items():
            # Extract features for prediction
            features = self._extract_worker_features(metrics, task)

            # Predict load and performance
            predicted_load = self._predict_worker_load(features)
            predicted_performance = self._predict_task_success(features, task)

            # Calculate composite score
            availability_score = max(0, 1.0 - (predicted_load / metrics.predicted_capacity))
            expertise_score = metrics.task_type_expertise.get(task.task_type, 0.5)
            efficiency_score = metrics.efficiency_score

            # Weighted combination
            composite_score = (
                availability_score * 0.4 +
                predicted_performance * 0.3 +
                expertise_score * 0.2 +
                efficiency_score * 0.1
            )

            worker_scores[worker_id] = composite_score

        # Select best worker
        if not worker_scores:
            return None

        best_worker = max(worker_scores.items(), key=self._get_worker_score)[0]

        # Update prediction cache
        predicted_load = self._predict_worker_load(
            self._extract_worker_features(self.worker_metrics[best_worker], task)
        )
        self.prediction_cache[best_worker] = (predicted_load, current_time)

        logger.info(f"AI selected worker {best_worker} with score {worker_scores[best_worker]:.3f}")
        return best_worker

    def _extract_worker_features(self, metrics: WorkerPerformanceMetrics, task: DistributedTask) -> np.ndarray:
        """Extract features for ML prediction"""
        if not SKLEARN_AVAILABLE:
            return np.array([metrics.current_load, metrics.performance_score])

        features = [
            metrics.current_load / max(1, metrics.max_capacity),  # Utilization
            metrics.performance_score,
            metrics.total_tasks,
            metrics.successful_tasks / max(1, metrics.total_tasks),  # Success rate
            metrics.avg_processing_time,
            time.time() - metrics.last_task_timestamp,  # Time since last task
            metrics.task_type_expertise.get(task.task_type, 0),
            len(metrics.recent_performance),
            metrics.efficiency_score,
            metrics.workload_trend,
            metrics.collaboration_score
        ]

        return np.array(features)

    def _predict_worker_load(self, features: np.ndarray) -> float:
        """Predict worker load using AI model"""
        if not SKLEARN_AVAILABLE or self.ai_model.load_predictor is None:
            # Fallback: simple heuristic
            return features[0] * 10  # Current utilization * capacity

        try:
            # Scale features
            features_scaled = self.ai_model.scaler.transform(features.reshape(1, -1))
            predicted_load = self.ai_model.load_predictor.predict(features_scaled)[0]
            return max(0, predicted_load)
        except Exception as e:
            logger.warning(f"Load prediction failed: {e}, using fallback")
            return features[0] * 10

    def _predict_task_success(self, features: np.ndarray, task: DistributedTask) -> float:
        """Predict task success probability"""
        if not SKLEARN_AVAILABLE or self.ai_model.performance_classifier is None:
            # Fallback: use historical success rate
            return features[3]  # Success rate feature

        try:
            features_scaled = self.ai_model.scaler.transform(features.reshape(1, -1))
            success_prob = self.ai_model.performance_classifier.predict(features_scaled)[0]
            return max(0, min(1, success_prob))
        except Exception as e:
            logger.warning(f"Success prediction failed: {e}, using fallback")
            return features[3]

    async def _train_ai_model(self):
        """Train AI models with historical data"""
        if not SKLEARN_AVAILABLE or len(self.task_history) < 100:
            return

        try:
            # Prepare training data
            X, y_load, y_performance = self._prepare_training_data()

            if len(X) < 50:
                logger.info("Insufficient training data for AI model")
                return

            # Scale features
            X_scaled = self.ai_model.scaler.fit_transform(X)

            # Train load predictor
            self.ai_model.load_predictor.fit(X_scaled, y_load)

            # Train performance classifier
            self.ai_model.performance_classifier.fit(X_scaled, y_performance)

            # Update training timestamp
            self.ai_model.last_training = time.time()

            # Calculate and store accuracy
            load_score = self.ai_model.load_predictor.score(X_scaled, y_load)
            perf_score = self.ai_model.performance_classifier.score(X_scaled, y_performance)
            self.ai_model.prediction_accuracy = (load_score + perf_score) / 2

            logger.info(f"AI model trained - Load accuracy: {load_score:.3f}, Performance accuracy: {perf_score:.3f}")

        except Exception as e:
            logger.error(f"AI model training failed: {e}")

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from task history"""
        X, y_load, y_performance = [], [], []

        for task_record in self.task_history[-500:]:  # Last 500 tasks
            if 'worker_features' in task_record and 'actual_load' in task_record:
                X.append(task_record['worker_features'])
                y_load.append(task_record['actual_load'])
                y_performance.append(1.0 if task_record.get('success', False) else 0.0)

        return np.array(X), np.array(y_load), np.array(y_performance)

    async def get_system_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive AI system performance metrics"""
        return {
            'ai_model_accuracy': self.ai_model.prediction_accuracy,
            'last_training': self.ai_model.last_training,
            'worker_count': len(self.worker_metrics),
            'total_tasks_processed': sum(m.total_tasks for m in self.worker_metrics.values()),
            'overall_success_rate': (
                sum(m.successful_tasks for m in self.worker_metrics.values()) /
                max(1, sum(m.total_tasks for m in self.worker_metrics.values()))
            ),
            'avg_processing_time': np.mean([m.avg_processing_time for m in self.worker_metrics.values()]) if self.worker_metrics else 0,
            'system_utilization': (
                sum(m.current_load for m in self.worker_metrics.values()) /
                max(1, sum(m.predicted_capacity for m in self.worker_metrics.values()))
            ),
            'prediction_cache_size': len(self.prediction_cache),
            'training_data_size': len(self.task_history),
            'sklearn_available': SKLEARN_AVAILABLE
        }

    def _get_task_ai_score(self, task):
        """Get AI score from task data"""
        return task.get('ai_score', 0)

    def _get_worker_score(self, worker_score_item):
        """Get score from worker score item tuple"""
        return worker_score_item[1]


async def shutdown_distributed_coordination():
    """Shutdown the global distributed task coordinator"""
    global coordinator
    if coordinator:
        await coordinator.stop()

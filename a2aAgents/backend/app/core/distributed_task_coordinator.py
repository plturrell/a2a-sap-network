"""
A2A-compliant distributed task coordination system
Provides Redis-based distributed task queue with failover and leader election
Integrates with A2A protocol, telemetry, and security systems
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from uuid import uuid4

# A2A SDK imports
from ..a2a.security.requestSigning import A2ARequestSigner

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
class DistributedTask:
    """Distributed task representation"""
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
        
        # Redis keys
        self.task_queue_key = f"{self.namespace}:queue"
        self.task_data_key = f"{self.namespace}:data"
        self.task_claims_key = f"{self.namespace}:claims"
        self.node_heartbeat_key = f"{self.namespace}:nodes"
    
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
        """Start worker processes"""
        self.running = True
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {num_workers} workers on node {self.node_id}")
    
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
        
        # Add to priority queue
        priority_score = self._get_priority_score(priority)
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
            stale_threshold = time.time() - 600
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
                            
                            # Re-add to queue
                            priority_score = self._get_priority_score(task.priority)
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


async def shutdown_distributed_coordination():
    """Shutdown the global distributed task coordinator"""
    global coordinator
    if coordinator:
        await coordinator.stop()
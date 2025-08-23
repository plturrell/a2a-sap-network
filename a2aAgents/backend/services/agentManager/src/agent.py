"""
Agent Manager - A2A Network Orchestrator
Manages agent discovery, health monitoring, workflow orchestration, and trust verification
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""


import os

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'shared')
sys.path.insert(0, shared_dir)

import sys
import os
# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"
    ROLLED_BACK = "rolled_back"


class WorkflowExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    SAGA = "saga"  # With compensation


@dataclass
class RegisteredAgent:
    """Registered A2A agent information"""
    agent_id: str
    name: str
    base_url: str
    capabilities: Dict[str, Any]
    status: AgentStatus = AgentStatus.UNKNOWN
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    health_check_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowInstance:
    """Active workflow instance"""
    workflow_id: str
    context_id: str
    status: WorkflowStatus
    agents_involved: List[str]
    current_agent: Optional[str]
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentManager(A2AAgentBase):
    """
    Agent Manager - Central orchestrator for A2A network
    """
    
    def __init__(self, base_url: str, redis_client: redis.Redis):
        super().__init__(
            agent_id="agent_manager",
            name="A2A Agent Manager",
            description="Central orchestrator for A2A network - handles discovery, monitoring, and workflow management",
            version="2.0.0",
            base_url=base_url
        )
        
        self.redis_client = redis_client
        self.registered_agents: Dict[str, RegisteredAgent] = {}
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.scheduler = AsyncIOScheduler()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.monitoring_active = False
        
        # Performance metrics
        self.metrics = {
            "total_agents_registered": 0,
            "total_workflows_processed": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "health_checks_performed": 0
        }
        
        # Track startup time
        self.started_at = datetime.utcnow()
        
        # Trust system components
        self.trust_keys: Dict[str, Dict[str, Any]] = {}  # agent_id -> {public_key, private_key, certificate}
        self.trust_store_path = os.getenv("TRUST_STORE_PATH", "/tmp/a2a_trust")
        
        # Circuit breakers for each agent
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize Agent Manager resources"""
        logger.info("Initializing Agent Manager...")
        
        # Initialize trust system
        self._initialize_trust_system()
        
        # Initialize HTTP client for agent communication
        self.http_client = None  # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        
        # Load registered agents from Redis
        await self._load_agents_from_redis()
        
        # Start scheduler for health monitoring
        self.scheduler.add_job(
            self._monitor_agent_health,
            IntervalTrigger(seconds=30),
            id="health_monitor",
            replace_existing=True
        )
        
        self.scheduler.add_job(
            self._cleanup_stale_workflows,
            IntervalTrigger(minutes=5),
            id="workflow_cleanup",
            replace_existing=True
        )
        
        self.scheduler.start()
        self.monitoring_active = True
        
        self.is_ready = True
        logger.info("Agent Manager initialized successfully")
    
    async def check_redis_connection(self) -> bool:
        """Check Redis connection health"""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    @a2a_handler("register_agent", "Register a new agent with the network")
    async def handle_agent_registration(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent registration requests"""
        try:
            registration_data = message.content if hasattr(message, 'content') else message
            
            # Validate registration data
            required_fields = ["agent_id", "name", "base_url", "capabilities"]
            for field in required_fields:
                if field not in registration_data:
                    return create_error_response(400, f"Missing required field: {field}")
            
            # Create registered agent
            agent = RegisteredAgent(
                agent_id=registration_data["agent_id"],
                name=registration_data["name"],
                base_url=registration_data["base_url"],
                capabilities=registration_data["capabilities"],
                status=AgentStatus.ACTIVE,
                last_heartbeat=datetime.utcnow(),
                metadata=registration_data.get("metadata", {})
            )
            
            # Store in memory and Redis
            self.registered_agents[agent.agent_id] = agent
            await self._save_agent_to_redis(agent)
            
            # Verify agent is reachable
            if await self._verify_agent_endpoint(agent):
                agent.status = AgentStatus.ACTIVE
            else:
                agent.status = AgentStatus.UNKNOWN
            
            # Generate trust credentials
            trust_credentials = await self._generate_trust_credentials(agent.agent_id)
            
            # Initialize circuit breaker for this agent
            self.circuit_breakers[agent.agent_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
            
            # Initialize agent with default reputation
            await self._initialize_agent_reputation(agent.agent_id)
            
            self.metrics["total_agents_registered"] += 1
            
            logger.info(f"Registered agent: {agent.agent_id} at {agent.base_url}")
            
            return create_success_response({
                "agent_id": agent.agent_id,
                "status": agent.status.value,
                "message": "Agent registered successfully",
                "trust_credentials": {
                    "trust_id": trust_credentials["trust_id"],
                    "public_key": trust_credentials["public_key"],
                    "certificate": trust_credentials["certificate"],
                    "expires_at": trust_credentials["expires_at"]
                }
            })
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("discover_agents", "Discover agents with specific capabilities")
    async def handle_agent_discovery(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent discovery requests"""
        try:
            query = message.content if hasattr(message, 'content') else message
            capability_filter = query.get("capabilities", {})
            status_filter = query.get("status", None)
            
            matching_agents = []
            
            for agent in self.registered_agents.values():
                # Filter by status
                if status_filter and agent.status.value != status_filter:
                    continue
                
                # Filter by capabilities
                if capability_filter:
                    agent_caps = agent.capabilities
                    match = True
                    for cap, value in capability_filter.items():
                        if cap not in agent_caps or agent_caps[cap] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                matching_agents.append({
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "base_url": agent.base_url,
                    "status": agent.status.value,
                    "capabilities": agent.capabilities
                })
            
            return create_success_response({
                "agents": matching_agents,
                "count": len(matching_agents)
            })
            
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("start_workflow", "Start a new A2A workflow")
    async def handle_workflow_start(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle workflow initiation"""
        try:
            workflow_def = message.content if hasattr(message, 'content') else message
            
            # Create workflow instance
            workflow = WorkflowInstance(
                workflow_id=f"wf_{context_id}_{int(time.time())}",
                context_id=context_id,
                status=WorkflowStatus.PENDING,
                agents_involved=workflow_def.get("agents", []),
                current_agent=workflow_def.get("start_agent"),
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=workflow_def.get("metadata", {})
            )
            
            # Validate agents exist
            for agent_id in workflow.agents_involved:
                if agent_id not in self.registered_agents:
                    return create_error_response(404, f"Agent not found: {agent_id}")
            
            # Store workflow
            self.active_workflows[workflow.workflow_id] = workflow
            await self._save_workflow_to_redis(workflow)
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow, workflow_def.get("initial_data", {})))
            
            self.metrics["total_workflows_processed"] += 1
            
            return create_success_response({
                "workflow_id": workflow.workflow_id,
                "status": workflow.status.value,
                "message": "Workflow started successfully"
            })
            
        except Exception as e:
            logger.error(f"Error starting workflow: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("get_network_status", "Get current A2A network status")
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        active_agents = [a for a in self.registered_agents.values() if a.status == AgentStatus.ACTIVE]
        
        return {
            "total_agents": len(self.registered_agents),
            "active_agents": len(active_agents),
            "unhealthy_agents": len([a for a in self.registered_agents.values() if a.status == AgentStatus.UNHEALTHY]),
            "active_workflows": len([w for w in self.active_workflows.values() if w.status == WorkflowStatus.RUNNING]),
            "completed_workflows": self.metrics["successful_workflows"],
            "failed_workflows": self.metrics["failed_workflows"],
            "uptime_seconds": int((datetime.utcnow() - self.started_at).total_seconds()) if hasattr(self, 'started_at') else 0
        }
    
    async def _monitor_agent_health(self):
        """Monitor health of all registered agents"""
        logger.debug("Running health check cycle")
        
        tasks = []
        for agent in list(self.registered_agents.values()):
            tasks.append(self._check_agent_health(agent))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.metrics["health_checks_performed"] += len(tasks)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _check_agent_health(self, agent: RegisteredAgent):
        """Check health of a single agent"""
        try:
            response = await self.http_client.get(
                f"{agent.base_url}/health",
                timeout=5.0
            )
            
            if response.status_code == 200:
                agent.status = AgentStatus.ACTIVE
                agent.last_heartbeat = datetime.utcnow()
                agent.health_check_failures = 0
            else:
                agent.health_check_failures += 1
                if agent.health_check_failures >= 3:
                    agent.status = AgentStatus.UNHEALTHY
                    
        except Exception as e:
            logger.warning(f"Health check failed for {agent.agent_id}: {e}")
            agent.health_check_failures += 1
            if agent.health_check_failures >= 3:
                agent.status = AgentStatus.UNHEALTHY
        
        # Update Redis
        await self._save_agent_to_redis(agent)
    
    async def _verify_agent_endpoint(self, agent: RegisteredAgent) -> bool:
        """Verify agent endpoint is accessible"""
        try:
            response = await self.http_client.get(
                f"{agent.base_url}/.well-known/agent.json",
                timeout=10.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def _execute_workflow(self, workflow: WorkflowInstance, initial_data: Dict[str, Any]):
        """Execute a workflow across agents"""
        try:
            workflow.status = WorkflowStatus.RUNNING
            await self._save_workflow_to_redis(workflow)
            
            current_data = initial_data
            
            # Execute workflow steps
            for agent_id in workflow.agents_involved:
                workflow.current_agent = agent_id
                workflow.updated_at = datetime.utcnow()
                
                agent = self.registered_agents.get(agent_id)
                if not agent or agent.status != AgentStatus.ACTIVE:
                    raise Exception(f"Agent {agent_id} not available")
                
                # Send data to agent
                result = await self._send_to_agent(agent, current_data, workflow.context_id)
                
                if result.get("status") == "error":
                    raise Exception(f"Agent {agent_id} returned error: {result.get('error')}")
                
                # Use result as input for next agent
                current_data = result.get("data", current_data)
                
                await self._save_workflow_to_redis(workflow)
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            self.metrics["successful_workflows"] += 1
            
        except Exception as e:
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = datetime.utcnow()
            self.metrics["failed_workflows"] += 1
        
        finally:
            await self._save_workflow_to_redis(workflow)
    
    async def _send_to_agent(self, agent: RegisteredAgent, data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Send data to an agent via A2A protocol"""
        try:
            rpc_request = {
                "jsonrpc": "2.0",
                "method": "process",
                "params": {
                    "data": data,
                    "context_id": context_id,
                    "source": self.agent_id
                },
                "id": f"{context_id}_{int(time.time())}"
            }
            
            response = await self.http_client.post(
                f"{agent.base_url}/a2a/{agent.agent_id}/v1/rpc",
                json=rpc_request,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    return {"status": "error", "error": result["error"]}
            
            return {"status": "error", "error": f"HTTP {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Error sending to agent {agent.agent_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _cleanup_stale_workflows(self):
        """Clean up old completed workflows"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        to_remove = []
        for workflow_id, workflow in self.active_workflows.items():
            if workflow.completed_at and workflow.completed_at < cutoff_time:
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self.active_workflows[workflow_id]
            await self.redis_client.delete(f"workflow:{workflow_id}")
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} stale workflows")
    
    async def _load_agents_from_redis(self):
        """Load registered agents from Redis"""
        try:
            keys = await self.redis_client.keys("agent:*")
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    agent_data = json.loads(data)
                    agent = RegisteredAgent(
                        agent_id=agent_data["agent_id"],
                        name=agent_data["name"],
                        base_url=agent_data["base_url"],
                        capabilities=agent_data["capabilities"],
                        status=AgentStatus(agent_data["status"]),
                        last_heartbeat=datetime.fromisoformat(agent_data["last_heartbeat"]),
                        health_check_failures=agent_data.get("health_check_failures", 0),
                        metadata=agent_data.get("metadata", {})
                    )
                    self.registered_agents[agent.agent_id] = agent
            
            logger.info(f"Loaded {len(self.registered_agents)} agents from Redis")
            
        except Exception as e:
            logger.error(f"Error loading agents from Redis: {e}")
    
    async def _save_agent_to_redis(self, agent: RegisteredAgent):
        """Save agent to Redis"""
        try:
            data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "base_url": agent.base_url,
                "capabilities": agent.capabilities,
                "status": agent.status.value,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "health_check_failures": agent.health_check_failures,
                "metadata": agent.metadata
            }
            await self.redis_client.set(
                f"agent:{agent.agent_id}",
                json.dumps(data),
                ex=86400  # Expire after 24 hours
            )
        except Exception as e:
            logger.error(f"Error saving agent to Redis: {e}")
    
    async def _save_workflow_to_redis(self, workflow: WorkflowInstance):
        """Save workflow to Redis"""
        try:
            data = {
                "workflow_id": workflow.workflow_id,
                "context_id": workflow.context_id,
                "status": workflow.status.value,
                "agents_involved": workflow.agents_involved,
                "current_agent": workflow.current_agent,
                "started_at": workflow.started_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "error": workflow.error,
                "metadata": workflow.metadata
            }
            await self.redis_client.set(
                f"workflow:{workflow.workflow_id}",
                json.dumps(data),
                ex=86400  # Expire after 24 hours
            )
        except Exception as e:
            logger.error(f"Error saving workflow to Redis: {e}")
    
    async def shutdown(self) -> None:
        """Cleanup Agent Manager resources"""
        logger.info("Shutting down Agent Manager...")
        
        self.scheduler.shutdown()
        self.monitoring_active = False
        
        if self.http_client:
            await self.http_client.aclose()
        
        self.is_ready = False
        logger.info("Agent Manager shutdown complete")
    
    def _initialize_trust_system(self):
        """Initialize the trust system with cryptographic keys"""
        os.makedirs(self.trust_store_path, exist_ok=True)
        
        # Generate or load agent manager's keys
        key_path = os.path.join(self.trust_store_path, "agent_manager_key.pem")
        if os.path.exists(key_path):
            # Load existing key
            with open(key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            # Generate new key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            # Save private key
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
        
        public_key = private_key.public_key()
        self.trust_keys["agent_manager"] = {
            "private_key": private_key,
            "public_key": public_key,
            "trust_id": f"trust_agent_manager_{hashlib.sha256(public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()[:16]}"
        }
        
        logger.info("Trust system initialized")
    
    async def _generate_trust_credentials(self, agent_id: str) -> Dict[str, Any]:
        """Generate trust credentials for an agent"""
        # Generate key pair for the agent
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Create a trust ID
        trust_id = f"trust_{agent_id}_{hashlib.sha256(public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()[:16]}"
        
        # Create a certificate (simplified - in production use proper X.509)
        certificate = {
            "trust_id": trust_id,
            "agent_id": agent_id,
            "public_key": base64.b64encode(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )).decode(),
            "issued_by": self.trust_keys["agent_manager"]["trust_id"],
            "issued_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat()
        }
        
        # Sign the certificate
        cert_data = json.dumps(certificate, sort_keys=True).encode()
        signature = self.trust_keys["agent_manager"]["private_key"].sign(
            cert_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        certificate["signature"] = base64.b64encode(signature).decode()
        
        # Store agent's public key
        self.trust_keys[agent_id] = {
            "public_key": public_key,
            "trust_id": trust_id,
            "certificate": certificate
        }
        
        return {
            "trust_id": trust_id,
            "private_key": base64.b64encode(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )).decode(),
            "public_key": certificate["public_key"],
            "certificate": base64.b64encode(json.dumps(certificate).encode()).decode(),
            "expires_at": certificate["expires_at"]
        }
    
    async def _initialize_agent_reputation(self, agent_id: str) -> None:
        """Initialize agent with default reputation in A2A Network"""
        try:
            # Call A2A Network service to create agent record with reputation
            a2a_network_url = "http://localhost:4004/api/v1/network"
            
            agent_data = {
                "address": f"0x{hashlib.sha256(agent_id.encode()).hexdigest()[:40]}",
                "name": agent_id,
                "endpoint": self.registered_agents[agent_id].base_url,
                "reputation": 100,  # Default reputation
                "isActive": True
            }
            
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{a2a_network_url}/Agents",
                    json=agent_data,
                    timeout=10.0
                )
                
                if response.status_code == 201:
                    logger.info(f"Agent {agent_id} registered in A2A Network with default reputation")
                    
                    # Initialize agent performance record
                    performance_data = {
                        "agent_ID": response.json()["ID"],
                        "totalTasks": 0,
                        "successfulTasks": 0,
                        "failedTasks": 0,
                        "reputationScore": 100,
                        "trustScore": 1.0
                    }
                    
                    await client.post(
                        f"{a2a_network_url}/AgentPerformance",
                        json=performance_data,
                        timeout=10.0
                    )
                    
                else:
                    logger.warning(f"Failed to register agent {agent_id} in A2A Network: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error initializing agent reputation for {agent_id}: {e}")
    
    async def update_agent_reputation(self, agent_id: str, reputation_change: int, reason: str) -> None:
        """Update agent reputation after task completion"""
        try:
            a2a_network_url = "http://localhost:4004/api/v1/network"
            
            # Apply reputation change through reputation service
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{a2a_network_url}/ReputationTransactions",
                    json={
                        "agent": {"ID": agent_id},
                        "transactionType": reason,
                        "amount": reputation_change,
                        "reason": reason,
                        "context": json.dumps({
                            "updated_by": "agent_manager",
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        "isAutomated": True
                    },
                    timeout=10.0
                )
                
                if response.status_code == 201:
                    logger.info(f"Reputation updated for agent {agent_id}: {reputation_change} points ({reason})")
                else:
                    logger.warning(f"Failed to update reputation for agent {agent_id}: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error updating agent reputation for {agent_id}: {e}")
    
    async def handle_task_completion(self, agent_id: str, task_result: Dict[str, Any]) -> None:
        """Handle task completion and update reputation accordingly"""
        try:
            # Calculate reputation change based on task result
            reputation_change = 0
            reason = "TASK_COMPLETION"
            
            if task_result.get("status") == "success":
                complexity = task_result.get("complexity", "MEDIUM")
                reputation_change = {
                    "SIMPLE": 5,
                    "MEDIUM": 10,
                    "COMPLEX": 20,
                    "CRITICAL": 30
                }.get(complexity, 10)
                
                # Performance bonuses
                if task_result.get("completion_time", 0) < task_result.get("expected_time", float('inf')) * 0.5:
                    reputation_change += 5  # Fast completion bonus
                    
                if task_result.get("accuracy", 0) > 0.95:
                    reputation_change += 10  # High accuracy bonus
                    
            elif task_result.get("status") == "failed":
                failure_type = task_result.get("failure_type", "ERROR")
                reputation_change = {
                    "TIMEOUT": -5,
                    "ERROR": -10,
                    "ABANDONED": -15
                }.get(failure_type, -10)
                reason = f"TASK_FAILURE_{failure_type}"
            
            # Update reputation
            if reputation_change != 0:
                await self.update_agent_reputation(agent_id, reputation_change, reason)
                
        except Exception as e:
            logger.error(f"Error handling task completion for agent {agent_id}: {e}")
    
    async def endorse_agent(self, endorser_id: str, endorsed_id: str, amount: int, reason: str) -> Dict[str, Any]:
        """Handle peer-to-peer agent endorsement"""
        try:
            a2a_network_url = "http://localhost:4004/api/v1/network"
            
            # Call the endorsement action on the A2A Network
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{a2a_network_url}/Agents({endorsed_id})/endorsePeer",
                    json={
                        "toAgentId": endorsed_id,
                        "amount": amount,
                        "reason": reason,
                        "description": f"Endorsed by {endorser_id}"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Agent {endorser_id} endorsed {endorsed_id} with {amount} points")
                    return create_success_response(result)
                else:
                    error_msg = f"Failed to endorse agent: {response.status_code}"
                    logger.warning(error_msg)
                    return create_error_response(response.status_code, error_msg)
                    
        except Exception as e:
            logger.error(f"Error endorsing agent {endorsed_id}: {e}")
            return create_error_response(500, str(e))


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.utcnow().timestamp() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow().timestamp()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
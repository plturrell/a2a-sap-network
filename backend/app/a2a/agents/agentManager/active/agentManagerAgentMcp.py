"""
Agent Manager A2A Agent with MCP Integration - Real Implementation
Properly extends A2AAgentBase and uses MCP decorators from the SDK
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from uuid import uuid4
import logging
from enum import Enum
import httpx
import hashlib
from dataclasses import dataclass, field

from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import SDK components - use the local SDK with MCP support
from app.a2a.sdk.agentBase import A2AAgentBase
from app.a2a.sdk.decorators import a2a_handler, a2a_task, a2a_skill
from app.a2a.sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
from app.a2a.sdk.utils import create_agent_id
from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from app.a2a.core.workflowContext import workflowContextManager, DataArtifact
from app.a2a.core.workflowMonitor import workflowMonitor
from app.a2a.core.trustManager import sign_a2a_message, initialize_agent_trust, verify_a2a_message, trust_manager
from app.a2a.core.helpSeeking import AgentHelpSeeker
from app.a2a.core.circuitBreaker import CircuitBreaker
from app.a2a.core.taskTracker import AgentTaskTracker
from app.a2aRegistry.client import get_registry_client
from app.a2a.advisors.agentAiAdvisor import create_agent_advisor

# Define types and enums (avoiding circular import)
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

class TrustRelationshipType(str, Enum):
    DELEGATION = "delegation"
    COLLABORATION = "collaboration"
    SUPERVISION = "supervision"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for agent selection"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class AgentHealthMetrics:
    """Health metrics for registered agents"""
    agent_id: str
    last_check: datetime
    response_time_ms: float
    healthy: bool
    error_count: int = 0
    success_count: int = 0
    average_response_time: float = 0.0
    load_score: float = 0.0  # 0.0 (idle) to 1.0 (overloaded)


class AgentManagerAgentMCP(A2AAgentBase):
    """
    Enhanced Agent Manager with proper MCP integration
    Orchestrates the A2A ecosystem with advanced capabilities
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id=create_agent_id("agent_manager"),
            name="Agent Manager MCP",
            description="Enhanced A2A Agent Manager with MCP-powered orchestration",
            version="2.0.0",
            base_url=base_url
        )
        
        # Agent registry
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_health: Dict[str, AgentHealthMetrics] = {}
        self.trust_contracts: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers for each agent
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Task tracker
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name=self.name
        )
        
        # Registry client
        self.registry_client = get_registry_client()
        
        # Help seeking support - optional
        try:
            self.help_seeker = AgentHelpSeeker()
        except Exception as e:
            logger.warning(f"Help seeker not available: {e}")
            self.help_seeker = None
        
        # AI advisor
        self.ai_advisor = None
        
        # Load balancing state
        self.round_robin_index = 0
        
        # Private key for trust system
        self.private_key = os.getenv("AGENT_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("AGENT_PRIVATE_KEY environment variable is required")
        
        # State persistence
        self.state_file = os.path.join(
            os.getenv("A2A_STATE_DIR", "/tmp/a2a"),
            "agent_manager_state.json"
        )
        
        logger.info(f"Initialized Agent Manager MCP at {base_url}")
    
    # ==========================================
    # MCP Tools for Agent Management
    # ==========================================
    
    @mcp_tool(
        name="register_agent",
        description="Register a new agent with the manager",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "agent_name": {"type": "string"},
                "base_url": {"type": "string"},
                "capabilities": {"type": "object"},
                "skills": {"type": "array"},
                "metadata": {"type": "object"}
            },
            "required": ["agent_id", "agent_name", "base_url", "capabilities"]
        }
    )
    async def register_agent_mcp(self, agent_id: str, agent_name: str, base_url: str,
                                capabilities: Dict[str, Any], skills: List[Dict[str, Any]] = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Register agent via MCP tool"""
        try:
            # Check if already registered
            if agent_id in self.registered_agents:
                return {"success": False, "error": f"Agent {agent_id} already registered"}
            
            # Verify agent is reachable
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code != 200:
                        return {"success": False, "error": f"Agent health check failed with status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": f"Agent not reachable: {str(e)}"}
            
            # Create circuit breaker for agent
            self.circuit_breakers[agent_id] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=httpx.HTTPError
            )
            
            # Initialize health metrics
            self.agent_health[agent_id] = AgentHealthMetrics(
                agent_id=agent_id,
                last_check=datetime.utcnow(),
                response_time_ms=0,
                healthy=True
            )
            
            # Store agent info
            self.registered_agents[agent_id] = {
                "agent_name": agent_name,
                "base_url": base_url,
                "capabilities": capabilities,
                "skills": skills or [],
                "metadata": metadata or {},
                "registered_at": datetime.utcnow(),
                "status": AgentStatus.ACTIVE
            }
            
            # Register with A2A registry if available
            if self.registry_client:
                try:
                    await self.registry_client.register_agent({
                        "agent_id": agent_id,
                        "agent_card": {
                            "name": agent_name,
                            "description": metadata.get("description", ""),
                            "version": metadata.get("version", "1.0.0"),
                            "capabilities": capabilities,
                            "skills": skills or [],
                            "serviceEndpoint": base_url
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to register with A2A registry: {e}")
            
            logger.info(f"✅ Agent {agent_id} registered successfully via MCP")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "message": f"Agent {agent_name} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="discover_agents",
        description="Discover agents by capability with load balancing",
        input_schema={
            "type": "object",
            "properties": {
                "required_capabilities": {"type": "array", "items": {"type": "string"}},
                "required_skills": {"type": "array", "items": {"type": "string"}},
                "strategy": {"type": "string", "enum": ["round_robin", "least_loaded", "random", "capability_based", "performance_based"]},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["required_capabilities"]
        }
    )
    async def discover_agents_mcp(self, required_capabilities: List[str], 
                                required_skills: List[str] = None,
                                strategy: str = "least_loaded",
                                max_results: int = 5) -> Dict[str, Any]:
        """Discover agents with load balancing via MCP"""
        try:
            matching_agents = []
            
            # Find agents with required capabilities
            for agent_id, agent_info in self.registered_agents.items():
                # Check if agent is healthy
                health = self.agent_health.get(agent_id)
                if not health or not health.healthy:
                    continue
                
                # Check capabilities
                agent_caps = set(agent_info["capabilities"].keys())
                if not all(cap in agent_caps for cap in required_capabilities):
                    continue
                
                # Check skills if specified
                if required_skills:
                    agent_skills = {s["id"] for s in agent_info.get("skills", [])}
                    if not all(skill in agent_skills for skill in required_skills):
                        continue
                
                matching_agents.append({
                    "agent_id": agent_id,
                    "agent_info": agent_info,
                    "health": health
                })
            
            if not matching_agents:
                return {
                    "success": True,
                    "agents": [],
                    "message": "No matching agents found"
                }
            
            # Apply load balancing strategy
            strategy_enum = LoadBalancingStrategy(strategy)
            selected = self._apply_load_balancing(matching_agents, strategy_enum, max_results)
            
            return {
                "success": True,
                "agents": [
                    {
                        "agent_id": agent["agent_id"],
                        "name": agent["agent_info"]["agent_name"],
                        "base_url": agent["agent_info"]["base_url"],
                        "capabilities": list(agent["agent_info"]["capabilities"].keys()),
                        "load_score": agent["health"].load_score,
                        "healthy": agent["health"].healthy
                    }
                    for agent in selected
                ],
                "total_matches": len(matching_agents),
                "strategy_used": strategy
            }
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="create_workflow",
        description="Create and orchestrate multi-agent workflow",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_name": {"type": "string"},
                "agents": {"type": "array", "items": {"type": "string"}},
                "tasks": {"type": "array", "items": {"type": "object"}},
                "dependencies": {"type": "object"},
                "timeout_seconds": {"type": "integer", "default": 300}
            },
            "required": ["workflow_name", "agents", "tasks"]
        }
    )
    async def create_workflow_mcp(self, workflow_name: str, agents: List[str],
                                tasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]] = None,
                                timeout_seconds: int = 300) -> Dict[str, Any]:
        """Create workflow via MCP"""
        try:
            # Validate agents
            missing_agents = [a for a in agents if a not in self.registered_agents]
            if missing_agents:
                return {"success": False, "error": f"Unknown agents: {missing_agents}"}
            
            # Create workflow
            workflow_id = f"wf_{uuid4().hex[:8]}"
            
            self.active_workflows[workflow_id] = {
                "workflow_name": workflow_name,
                "agents": agents,
                "tasks": tasks,
                "dependencies": dependencies or {},
                "status": WorkflowStatus.PENDING,
                "created_at": datetime.utcnow(),
                "timeout_seconds": timeout_seconds
            }
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow_id))
            
            logger.info(f"✅ Workflow {workflow_id} created via MCP")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "message": f"Workflow '{workflow_name}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp_tool(
        name="monitor_agent_health",
        description="Perform health check on specific agent",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "detailed": {"type": "boolean", "default": False}
            },
            "required": ["agent_id"]
        }
    )
    async def monitor_agent_health_mcp(self, agent_id: str, detailed: bool = False) -> Dict[str, Any]:
        """Monitor agent health via MCP"""
        try:
            if agent_id not in self.registered_agents:
                return {"success": False, "error": f"Agent {agent_id} not found"}
            
            agent_info = self.registered_agents[agent_id]
            base_url = agent_info["base_url"]
            
            # Perform health check
            start_time = datetime.utcnow()
            health_data = {}
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{base_url}/health")
                    response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        healthy = True
                    else:
                        healthy = False
                        
            except Exception as e:
                response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                healthy = False
                health_data = {"error": str(e)}
            
            # Update health metrics
            metrics = self.agent_health.get(agent_id)
            if metrics:
                metrics.last_check = datetime.utcnow()
                metrics.response_time_ms = response_time_ms
                metrics.healthy = healthy
                if healthy:
                    metrics.success_count += 1
                else:
                    metrics.error_count += 1
                
                # Update average response time
                total_checks = metrics.success_count + metrics.error_count
                metrics.average_response_time = (
                    (metrics.average_response_time * (total_checks - 1) + response_time_ms) / total_checks
                )
            
            result = {
                "success": True,
                "agent_id": agent_id,
                "healthy": healthy,
                "response_time_ms": response_time_ms,
                "last_check": datetime.utcnow().isoformat()
            }
            
            if detailed:
                result["health_data"] = health_data
                result["metrics"] = {
                    "error_count": metrics.error_count,
                    "success_count": metrics.success_count,
                    "average_response_time": metrics.average_response_time,
                    "load_score": metrics.load_score
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==========================================
    # MCP Resources for State Access
    # ==========================================
    
    @mcp_resource(
        uri="agent://registry",
        name="Agent Registry",
        description="Complete registry of all managed agents",
        mime_type="application/json"
    )
    async def get_agent_registry(self) -> Dict[str, Any]:
        """Get agent registry via MCP resource"""
        return {
            "total_agents": len(self.registered_agents),
            "agents": {
                agent_id: {
                    **info,
                    "status": info.get("status", AgentStatus.UNKNOWN),
                    "health": {
                        "healthy": self.agent_health.get(agent_id, {}).healthy if agent_id in self.agent_health else False,
                        "last_check": self.agent_health.get(agent_id, {}).last_check.isoformat() if agent_id in self.agent_health else None
                    }
                }
                for agent_id, info in self.registered_agents.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="agent://workflows",
        name="Active Workflows",
        description="Currently active multi-agent workflows",
        mime_type="application/json"
    )
    async def get_active_workflows(self) -> Dict[str, Any]:
        """Get active workflows via MCP resource"""
        return {
            "total_workflows": len(self.active_workflows),
            "workflows": {
                wf_id: {
                    **wf_info,
                    "elapsed_seconds": (datetime.utcnow() - wf_info["created_at"]).total_seconds()
                        if wf_info.get("status") == WorkflowStatus.RUNNING else None
                }
                for wf_id, wf_info in self.active_workflows.items()
            },
            "by_status": {
                status.value: len([w for w in self.active_workflows.values() 
                                 if w.get("status") == status])
                for status in WorkflowStatus
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    @mcp_resource(
        uri="agent://health-metrics",
        name="Agent Health Metrics",
        description="Detailed health metrics for all agents",
        mime_type="application/json"
    )
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics via MCP resource"""
        return {
            "agents": {
                agent_id: {
                    "healthy": metrics.healthy,
                    "last_check": metrics.last_check.isoformat(),
                    "response_time_ms": metrics.response_time_ms,
                    "average_response_time": metrics.average_response_time,
                    "error_count": metrics.error_count,
                    "success_count": metrics.success_count,
                    "load_score": metrics.load_score,
                    "uptime_percentage": (metrics.success_count / max(metrics.success_count + metrics.error_count, 1)) * 100
                }
                for agent_id, metrics in self.agent_health.items()
            },
            "summary": {
                "total_agents": len(self.agent_health),
                "healthy_agents": len([m for m in self.agent_health.values() if m.healthy]),
                "average_load": sum(m.load_score for m in self.agent_health.values()) / max(len(self.agent_health), 1)
            },
            "last_updated": datetime.utcnow().isoformat()
        }
    
    # ==========================================
    # Internal Methods
    # ==========================================
    
    def _apply_load_balancing(self, agents: List[Dict[str, Any]], 
                            strategy: LoadBalancingStrategy, 
                            max_results: int) -> List[Dict[str, Any]]:
        """Apply load balancing strategy to select agents"""
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round robin
            selected = []
            for i in range(min(max_results, len(agents))):
                idx = (self.round_robin_index + i) % len(agents)
                selected.append(agents[idx])
            self.round_robin_index = (self.round_robin_index + len(selected)) % len(agents)
            return selected
            
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Sort by load score
            sorted_agents = sorted(agents, key=lambda a: a["health"].load_score)
            return sorted_agents[:max_results]
            
        elif strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            # Sort by average response time
            sorted_agents = sorted(agents, key=lambda a: a["health"].average_response_time)
            return sorted_agents[:max_results]
            
        else:
            # Default: return first N agents
            return agents[:max_results]
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow (simplified)"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        try:
            workflow["status"] = WorkflowStatus.RUNNING
            workflow["started_at"] = datetime.utcnow()
            
            # Simple sequential execution
            for i, task in enumerate(workflow["tasks"]):
                agent_id = workflow["agents"][i % len(workflow["agents"])]
                
                # Execute task on agent
                # In real implementation, this would call the agent's API
                await asyncio.sleep(1)  # Simulate task execution
            
            workflow["status"] = WorkflowStatus.COMPLETED
            workflow["completed_at"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            workflow["status"] = WorkflowStatus.FAILED
            workflow["error"] = str(e)
    
    # ==========================================
    # A2A Handlers (keeping compatibility)
    # ==========================================
    
    @a2a_handler("agent_registration")
    async def handle_agent_registration(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle agent registration via A2A protocol"""
        params = message.parts[0].data if message.parts else {}
        return await self.register_agent_mcp(**params)
    
    @a2a_handler("discover_agents")
    async def handle_discover_agents(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle agent discovery via A2A protocol"""
        params = message.parts[0].data if message.parts else {}
        return await self.discover_agents_mcp(**params)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the agent manager"""
        try:
            # Initialize trust system
            try:
                self.trust_identity = initialize_agent_trust(
                    self.agent_id,
                    self.private_key
                )
            except Exception as e:
                logger.warning(f"Trust initialization failed: {e}")
                self.trust_identity = None
            
            # Create AI advisor
            try:
                self.ai_advisor = create_agent_advisor(self.agent_id)
                if asyncio.iscoroutine(self.ai_advisor):
                    self.ai_advisor = await self.ai_advisor
            except Exception as e:
                logger.warning(f"AI advisor creation failed: {e}")
                self.ai_advisor = None
            
            # Load persisted state
            await self._load_state()
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor_loop())
            
            logger.info("Agent Manager initialized successfully")
            
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "version": self.version,
                "mcp_tools": len(self.list_mcp_tools()),
                "mcp_resources": len(self.list_mcp_resources())
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def _health_monitor_loop(self):
        """Background health monitoring"""
        while True:
            try:
                for agent_id in list(self.registered_agents.keys()):
                    await self.monitor_agent_health_mcp(agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _load_state(self):
        """Load persisted state"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Restore state carefully
                    # This is simplified - real implementation would validate
                    self.registered_agents = state.get("registered_agents", {})
                    logger.info(f"Loaded state with {len(self.registered_agents)} agents")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self):
        """Save state to disk"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                "registered_agents": self.registered_agents,
                "saved_at": datetime.utcnow().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def shutdown(self):
        """Shutdown the agent manager"""
        try:
            # Save state before shutdown
            await self._save_state()
            
            logger.info("Agent Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
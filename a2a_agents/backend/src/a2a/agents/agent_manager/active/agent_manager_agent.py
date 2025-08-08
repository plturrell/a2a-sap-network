"""
Agent Manager A2A Agent - The orchestrator of the A2A ecosystem
Handles agent registration, trust contracts, and workflow management
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

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..core.a2a_types import A2AMessage, MessagePart, MessageRole
from .data_standardization_agent import TaskState, TaskStatus, TaskArtifact, AgentCard
from src.a2a.core.workflow_context import workflow_context_manager, DataArtifact
from src.a2a.core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import initialize_agent_trust, sign_a2a_message, get_trust_contract, verify_a2a_message
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage, create_delegation_contract
from app.a2a_registry.client import get_registry_client
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus
from ..core.circuit_breaker import CircuitBreaker, CircuitState

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


class TrustRelationshipType(str, Enum):
    DELEGATION = "delegation"
    COLLABORATION = "collaboration"
    SUPERVISION = "supervision"


class AgentRegistrationRequest(BaseModel):
    agent_id: str
    agent_name: str
    base_url: str
    capabilities: Dict[str, Any]
    skills: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class TrustContractRequest(BaseModel):
    delegator_agent: str
    delegate_agent: str
    actions: List[str]
    expiry_hours: Optional[int] = 24
    conditions: Optional[Dict[str, Any]] = None


class WorkflowRequest(BaseModel):
    workflow_name: str
    agents: List[str]
    tasks: List[Dict[str, Any]]
    dependencies: Optional[Dict[str, List[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentManagerCard(BaseModel):
    name: str = "Agent Manager"
    description: str = "Manages A2A ecosystem registration, trust contracts, and workflow orchestration"
    url: str
    version: str = "2.0.0"
    protocolVersion: str = "0.2.9"
    provider: Dict[str, str] = {
        "organization": "FinSight CIB",
        "url": "https://finsight-cib.com"
    }
    capabilities: Dict[str, bool] = {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
        "batchProcessing": True,
        "smartContractDelegation": True,
        "aiAdvisor": True,
        "helpSeeking": True,
        "taskTracking": True,
        "agentRegistration": True,
        "trustContractManagement": True,
        "workflowOrchestration": True,
        "systemMonitoring": True,
        "circuitBreaker": True
    }
    defaultInputModes: List[str] = ["application/json", "text/plain"]
    defaultOutputModes: List[str] = ["application/json"]
    skills: List[Dict[str, Any]] = [
        {
            "id": "agent-registration",
            "name": "Agent Registration Management",
            "description": "Register, deregister, and manage A2A agents in the ecosystem",
            "tags": ["registration", "management", "lifecycle"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "trust-contract-management",
            "name": "Trust Contract Management", 
            "description": "Create, update, and validate smart contracts and delegation relationships",
            "tags": ["trust", "security", "contracts"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "workflow-orchestration",
            "name": "Workflow Orchestration",
            "description": "Coordinate multi-agent workflows and task distribution",
            "tags": ["orchestration", "workflow", "coordination"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "system-monitoring",
            "name": "System Monitoring",
            "description": "Monitor agent health, performance, and system metrics",
            "tags": ["monitoring", "health", "metrics"],
            "inputModes": ["application/json"], 
            "outputModes": ["application/json"]
        }
    ]


class AgentManagerAgent(AgentHelpSeeker):
    """Agent Manager - Orchestrates A2A ecosystem registration, trust, and workflows"""
    
    def __init__(self, base_url: str, agent_id: str = "agent_manager", agent_name: str = "Agent Manager", 
                 capabilities: Optional[Dict[str, Any]] = None, skills: Optional[List[Dict[str, Any]]] = None):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        self.registry_client = None
        self.capabilities = capabilities or {}
        self.skills = skills or []
        
        # Initialize smart contract trust identity
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_identity = initialize_agent_trust(
            self.agent_id,
            agent_name
        )
        logger.info(f"✅ Agent Manager trust identity initialized: {self.agent_id}")
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name=self.agent_name
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "role": "ecosystem_orchestrator",
            "capabilities": ["agent_management", "trust_contracts", "workflow_orchestration"]
        }
        
        self.help_action_engine = None
        try:
            from ..core.help_action_engine import HelpActionEngine
            self.help_action_engine = HelpActionEngine(
                agent_id=self.agent_id,
                agent_context=agent_context
            )
            logger.info(f"✅ Help action system initialized for {self.agent_id}")
        except Exception as e:
            logger.warning(f"Help action system initialization failed: {e}")
        
        # Agent state tracking with persistence
        self.registered_agents = {}  # agent_id -> agent_info
        self.trust_contracts = {}   # contract_id -> contract_info
        self.active_workflows = {}  # workflow_id -> workflow_info
        self.agent_health_cache = {}  # agent_id -> last_health_check
        self.startup_time = datetime.utcnow()  # Track when agent started
        
        # Initialize persistent storage
        self._storage_path = os.getenv("AGENT_MANAGER_STORAGE_PATH", "/tmp/agent_manager_state")
        os.makedirs(self._storage_path, exist_ok=True)
        
        # Initialize circuit breakers for agent communication
        self.circuit_breakers = {}  # agent_id -> CircuitBreaker
        
        # Load persisted state
        asyncio.create_task(self._load_persisted_state())
        
        # Initialize message queue with agent-specific configuration
        self.initialize_message_queue(
            agent_id=self.agent_id,
            max_concurrent_processing=5,
            auto_mode_threshold=8,
            enable_streaming=True,
            enable_batch_processing=True
        )
        
        # Set message processor callback
        self.message_queue.set_message_processor(self._process_message_core)
        
        # Initialize registry client for service discovery
        try:
            self.registry_client = get_registry_client()
            logger.info("✅ A2A Registry client initialized for Agent Manager")
        except Exception as e:
            logger.warning(f"Failed to initialize registry client: {e}")
        
        # Initialize Database-backed AI Decision Logger
        from ..core.ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
        
        # Construct Data Manager URL
        data_manager_url = f"{self.base_url.replace('/agents/', '/').rstrip('/')}/data-manager"
        
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=self.agent_id,
            data_manager_url=data_manager_url,
            memory_size=1500,  # Higher memory for ecosystem management decisions
            learning_threshold=12,  # Higher threshold due to complexity
            cache_ttl=600  # Longer cache for management decisions
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI advisor
        self.ai_advisor = create_agent_advisor(
            agent_name="Agent Manager",
            agent_description="A2A ecosystem orchestrator managing agent registration, trust contracts, and workflows",
            skills=["agent_management", "trust_contracts", "workflow_orchestration", "system_monitoring"],
            knowledge_base=self._initialize_advisor_knowledge()
        )
        logger.info(f"✅ AI Advisor initialized for Agent Manager")
        
        logger.info(f"✅ Agent Manager v2.0.0 initialized with full orchestration capabilities")
    
    def _initialize_advisor_knowledge(self) -> List[Dict[str, str]]:
        """Initialize AI advisor knowledge base"""
        return [
            {
                "question": "How do I register a new agent in the A2A ecosystem?",
                "answer": "Send a POST request to /agents/register with agent_id, name, base_url, capabilities, and skills. The Agent Manager will validate the agent, create trust contracts, and register it in the ecosystem.",
                "tags": ["registration", "onboarding"]
            },
            {
                "question": "How are trust contracts managed?", 
                "answer": "Trust contracts are created via /trust/contracts endpoint. Specify delegator_agent, delegate_agent, actions, and expiry. The Agent Manager validates permissions and creates smart contracts for secure delegation.",
                "tags": ["trust", "security", "delegation"]
            },
            {
                "question": "How do I orchestrate a multi-agent workflow?",
                "answer": "Use the /workflows endpoint to create workflows with agent lists, tasks, and dependencies. The Agent Manager will coordinate execution, monitor progress, and handle failures.",
                "tags": ["workflow", "orchestration", "coordination"]
            },
            {
                "question": "How can I monitor agent health?",
                "answer": "Use /agents/health endpoint for individual agents or /system/health for ecosystem overview. The Agent Manager continuously monitors all registered agents.",
                "tags": ["monitoring", "health", "diagnostics"]
            },
            {
                "question": "What happens when an agent fails?",
                "answer": "The Agent Manager detects failures through health checks, attempts recovery, redistributes tasks if possible, and notifies dependent agents. It maintains workflow continuity.",
                "tags": ["failure", "recovery", "resilience"]
            }
        ]
    
    async def get_agent_card(self) -> AgentManagerCard:
        """Get the agent card for Agent Manager"""
        return AgentManagerCard(url=self.base_url)
    
    async def process_message(
        self, 
        message: A2AMessage, 
        context_id: str,
        priority: str = "medium",
        processing_mode: str = "auto"
    ) -> Dict[str, Any]:
        """Process A2A message with queue support (streaming or batched)"""
        # Check if this is an AI advisor request first
        if self._is_advisor_request(message):
            return await self._handle_advisor_request(message)
        
        # Handle the message content processing
        if hasattr(message, 'model_dump'):
            message_data = message.model_dump()
        elif hasattr(message, 'dict'):
            message_data = message.dict()
        else:
            message_data = {"content": str(message)}
        
        # Convert string priority to enum
        from ..core.message_queue import MessagePriority, ProcessingMode
        try:
            msg_priority = MessagePriority(priority.lower())
        except ValueError:
            msg_priority = MessagePriority.MEDIUM
            
        try:
            proc_mode = ProcessingMode(processing_mode.lower())
        except ValueError:
            proc_mode = ProcessingMode.AUTO
        
        # Enqueue message for processing
        message_id = await self.message_queue.enqueue_message(
            a2a_message=message.model_dump(),
            context_id=context_id,
            priority=msg_priority,
            processing_mode=proc_mode
        )
        
        # For immediate/streaming mode, we need to wait for the result
        if proc_mode == ProcessingMode.IMMEDIATE or (proc_mode == ProcessingMode.AUTO and 
                                                     len(self.message_queue._processing) + len(self.message_queue._messages) < self.message_queue.auto_mode_threshold):
            # Wait for completion
            max_wait = 30  # 30 seconds max wait
            wait_interval = 0.1
            waited = 0
            
            while waited < max_wait:
                msg_status = self.message_queue.get_message_status(message_id)
                if msg_status and msg_status.get("status") in ["completed", "failed", "timeout"]:
                    return msg_status.get("result", {"error": "No result available"})
                await asyncio.sleep(wait_interval)
                waited += wait_interval
            
            return {"error": "Message processing timeout", "message_id": message_id}
        else:
            # For queued mode, return immediately with message ID
            return {
                "message_type": "queued_for_processing",
                "message_id": message_id,
                "queue_position": self.message_queue.stats.queue_depth,
                "estimated_processing_time": self.message_queue.stats.avg_processing_time
            }
    
    async def _process_message_core(self, message: A2AMessage, context_id: str) -> A2AMessage:
        """Process incoming A2A message"""
        # Check if this is an AI advisor request
        if self._is_advisor_request(message):
            return await self._handle_advisor_request(message)
        
        # Verify message trust
        trust_score = verify_a2a_message(message.model_dump(), message.signature)
        if trust_score < 0.5:
            logger.warning(f"⚠️ Low trust score {trust_score} for message {message.messageId}")
        
        # Extract request from message
        request_data = await self._extract_management_request(message)
        operation = request_data.get("operation")
        
        # Route to appropriate handler
        try:
            if operation == "register_agent":
                result = await self._handle_agent_registration(request_data, context_id)
            elif operation == "deregister_agent":
                result = await self._handle_agent_deregistration(request_data, context_id)
            elif operation == "create_trust_contract":
                result = await self._handle_trust_contract_creation(request_data, context_id)
            elif operation == "create_workflow":
                result = await self._handle_workflow_creation(request_data, context_id)
            elif operation == "monitor_agents":
                result = await self._handle_agent_monitoring(request_data, context_id)
            elif operation == "system_health":
                result = await self._handle_system_health_check(request_data, context_id)
            elif operation == "discover_agents":
                result = await self._handle_agent_discovery(request_data, context_id)
            else:
                result = {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            logger.error(f"Error processing Agent Manager operation {operation}: {e}")
            # Use help-seeking for complex errors
            await self._handle_error_with_help_seeking(e, message.taskId or str(uuid4()), context_id)
            result = {"error": str(e)}
        
        # Create response
        response = A2AMessage(
            role=MessageRole.AGENT,
            taskId=message.taskId,
            contextId=context_id,
            parts=[
                MessagePart(
                    kind="text", 
                    text=json.dumps(result)
                )
            ]
        )
        
        # Sign response
        response.signature = sign_a2a_message(response.model_dump())
        
        return response
    
    async def _extract_management_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract management request from A2A message"""
        for part in message.parts:
            if part.kind == "text" and part.text:
                try:
                    return json.loads(part.text)
                except json.JSONDecodeError:
                    return {"operation": "unknown", "error": "Invalid JSON"}
            elif part.kind == "data" and part.data:
                return part.data
        
        return {"operation": "unknown", "error": "No valid request data found"}
    
    async def _handle_agent_registration(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle agent registration request"""
        try:
            agent_id = request_data.get("agent_id")
            agent_name = request_data.get("agent_name")
            base_url = request_data.get("base_url")
            capabilities = request_data.get("capabilities", {})
            skills = request_data.get("skills", [])
            
            if not all([agent_id, agent_name, base_url]):
                return {"success": False, "error": "Missing required fields: agent_id, agent_name, base_url"}
            
            # Check if agent already registered
            if agent_id in self.registered_agents:
                return {"success": False, "error": f"Agent {agent_id} already registered"}
            
            # Validate agent health
            agent_health = await self._check_agent_health(base_url)
            if not agent_health.get("healthy"):
                return {"success": False, "error": f"Agent at {base_url} is not responding to health checks"}
            
            # Initialize trust contract for new agent
            trust_contract_id = await self._create_agent_trust_contract(agent_id, agent_name)
            
            # Register with A2A registry
            registration_id = None
            if self.registry_client:
                try:
                    agent_info = await self.registry_client.register_agent(
                        agent_card={
                            "name": agent_name,
                            "url": base_url,
                            "capabilities": capabilities,
                            "skills": skills,
                            "version": "2.0.0",
                            "protocolVersion": "0.2.9"
                        },
                        agent_id=agent_id
                    )
                    registration_id = agent_info.get("id")
                except Exception as e:
                    logger.warning(f"Failed to register {agent_id} with A2A registry: {e}")
            
            # Store agent information
            self.registered_agents[agent_id] = {
                "agent_name": agent_name,
                "base_url": base_url,
                "capabilities": capabilities,
                "skills": skills,
                "registration_id": registration_id,
                "trust_contract_id": trust_contract_id,
                "registered_at": datetime.utcnow(),
                "last_health_check": datetime.utcnow(),
                "status": AgentStatus.ACTIVE
            }
            
            # Persist state immediately after registration
            asyncio.create_task(self._persist_state())
            
            # Create workflow context for registration
            workflow_context_manager.create_workflow_context(
                workflow_id=f"agent_registration_{agent_id}",
                context_id=context_id,
                metadata={
                    "operation": "agent_registration",
                    "agent_id": agent_id,
                    "agent_name": agent_name
                }
            )
            
            logger.info(f"✅ Agent {agent_id} registered successfully")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "registration_id": registration_id,
                "trust_contract_id": trust_contract_id,
                "message": f"Agent {agent_name} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_agent_deregistration(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle agent deregistration request"""
        try:
            agent_id = request_data.get("agent_id")
            force = request_data.get("force", False)
            
            if not agent_id:
                return {"success": False, "error": "Missing required field: agent_id"}
            
            if agent_id not in self.registered_agents:
                return {"success": False, "error": f"Agent {agent_id} not found"}
            
            agent_info = self.registered_agents[agent_id]
            
            # Check for active workflows unless forced
            if not force:
                active_workflows = [wf for wf in self.active_workflows.values() 
                                  if agent_id in wf.get("agents", []) and wf.get("status") == WorkflowStatus.RUNNING]
                if active_workflows:
                    return {"success": False, "error": f"Agent {agent_id} has {len(active_workflows)} active workflows. Use force=true to override."}
            
            # Deregister from A2A registry
            if self.registry_client and agent_info.get("registration_id"):
                try:
                    await self.registry_client.deregister_agent(agent_id)
                except Exception as e:
                    logger.warning(f"Failed to deregister {agent_id} from A2A registry: {e}")
            
            # Remove trust contracts
            trust_contract_id = agent_info.get("trust_contract_id")
            if trust_contract_id and trust_contract_id in self.trust_contracts:
                del self.trust_contracts[trust_contract_id]
            
            # Remove from registered agents
            del self.registered_agents[agent_id]
            
            # Persist state after deregistration
            asyncio.create_task(self._persist_state())
            
            logger.info(f"✅ Agent {agent_id} deregistered successfully")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "message": f"Agent {agent_id} deregistered successfully"
            }
            
        except Exception as e:
            logger.error(f"Agent deregistration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_trust_contract_creation(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle trust contract creation request"""
        try:
            delegator_agent = request_data.get("delegator_agent")
            delegate_agent = request_data.get("delegate_agent") 
            actions = request_data.get("actions", [])
            expiry_hours = request_data.get("expiry_hours", 24)
            conditions = request_data.get("conditions", {})
            
            if not all([delegator_agent, delegate_agent, actions]):
                return {"success": False, "error": "Missing required fields: delegator_agent, delegate_agent, actions"}
            
            # Validate both agents are registered
            if delegator_agent not in self.registered_agents:
                return {"success": False, "error": f"Delegator agent {delegator_agent} not registered"}
            if delegate_agent not in self.registered_agents:
                return {"success": False, "error": f"Delegate agent {delegate_agent} not registered"}
            
            # Create delegation contract
            try:
                contract_id = f"trust_{delegator_agent}_{delegate_agent}_{uuid4().hex[:8]}"
                expiry_time = datetime.utcnow() + timedelta(hours=expiry_hours)
                
                # Validate and convert actions to DelegationAction enums
                validated_actions = []
                for action in actions:
                    try:
                        # Try to match action string to DelegationAction enum values
                        action_upper = action.upper()
                        for delegation_action in DelegationAction:
                            if delegation_action.value.upper() == action_upper or delegation_action.name == action_upper:
                                validated_actions.append(delegation_action)
                                break
                        else:
                            logger.warning(f"Invalid delegation action: {action}")
                    except Exception as e:
                        logger.warning(f"Failed to validate action {action}: {e}")
                
                if not validated_actions:
                    return {"success": False, "error": f"No valid delegation actions provided from: {actions}"}
                
                # Use delegation contract system
                delegation_contract = create_delegation_contract(
                    delegator_agent=delegator_agent,
                    delegate_agent=delegate_agent,
                    actions=validated_actions,
                    expiry_time=expiry_time,
                    conditions=conditions
                )
                
                # Store contract info
                self.trust_contracts[contract_id] = {
                    "delegator_agent": delegator_agent,
                    "delegate_agent": delegate_agent,
                    "actions": actions,
                    "expiry_time": expiry_time,
                    "conditions": conditions,
                    "created_at": datetime.utcnow(),
                    "status": "active"
                }
                
                # Persist state after contract creation
                asyncio.create_task(self._persist_state())
                
                logger.info(f"✅ Trust contract {contract_id} created successfully")
                
                return {
                    "success": True,
                    "contract_id": contract_id,
                    "expiry_time": expiry_time.isoformat(),
                    "message": f"Trust contract created between {delegator_agent} and {delegate_agent}"
                }
                
            except Exception as e:
                logger.error(f"Failed to create delegation contract: {e}")
                return {"success": False, "error": f"Contract creation failed: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Trust contract creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_workflow_creation(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle workflow creation and orchestration"""
        try:
            workflow_name = request_data.get("workflow_name")
            agents = request_data.get("agents", [])
            tasks = request_data.get("tasks", [])
            dependencies = request_data.get("dependencies", {})
            metadata = request_data.get("metadata", {})
            
            if not all([workflow_name, agents, tasks]):
                return {"success": False, "error": "Missing required fields: workflow_name, agents, tasks"}
            
            # Validate all agents are registered
            unregistered_agents = [agent for agent in agents if agent not in self.registered_agents]
            if unregistered_agents:
                return {"success": False, "error": f"Unregistered agents: {unregistered_agents}"}
            
            # Create workflow
            workflow_id = f"workflow_{uuid4().hex[:8]}"
            
            workflow_info = {
                "workflow_name": workflow_name,
                "workflow_id": workflow_id,
                "agents": agents,
                "tasks": tasks,
                "dependencies": dependencies,
                "metadata": metadata,
                "status": WorkflowStatus.PENDING,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "completed_at": None,
                "progress": {"completed": 0, "total": len(tasks)}
            }
            
            self.active_workflows[workflow_id] = workflow_info
            
            # Create workflow context
            workflow_context_manager.create_workflow_context(
                workflow_id=workflow_id,
                context_id=context_id,
                metadata={
                    "workflow_name": workflow_name,
                    "agents": agents,
                    "task_count": len(tasks)
                }
            )
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow_id))
            
            logger.info(f"✅ Workflow {workflow_id} created and started")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": WorkflowStatus.PENDING,
                "message": f"Workflow {workflow_name} created and started"
            }
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_agent_monitoring(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle agent monitoring request"""
        try:
            agent_id = request_data.get("agent_id")
            
            if agent_id:
                # Monitor specific agent
                if agent_id not in self.registered_agents:
                    return {"success": False, "error": f"Agent {agent_id} not registered"}
                
                agent_info = self.registered_agents[agent_id]
                health = await self._check_agent_health(agent_info["base_url"])
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "status": agent_info["status"],
                    "health": health,
                    "last_health_check": agent_info["last_health_check"].isoformat()
                }
            else:
                # Monitor all agents
                agent_statuses = {}
                for aid, info in self.registered_agents.items():
                    health = await self._check_agent_health(info["base_url"])
                    agent_statuses[aid] = {
                        "status": info["status"],
                        "health": health,
                        "last_health_check": info["last_health_check"].isoformat()
                    }
                
                return {
                    "success": True,
                    "agent_count": len(self.registered_agents),
                    "agents": agent_statuses
                }
                
        except Exception as e:
            logger.error(f"Agent monitoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_system_health_check(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle system-wide health check"""
        try:
            # Check all registered agents
            healthy_agents = 0
            unhealthy_agents = 0
            
            for agent_id, agent_info in self.registered_agents.items():
                health = await self._check_agent_health(agent_info["base_url"])
                if health.get("healthy"):
                    healthy_agents += 1
                    agent_info["status"] = AgentStatus.ACTIVE
                else:
                    unhealthy_agents += 1
                    agent_info["status"] = AgentStatus.UNHEALTHY
                
                agent_info["last_health_check"] = datetime.utcnow()
            
            # Check active workflows
            running_workflows = len([wf for wf in self.active_workflows.values() if wf["status"] == WorkflowStatus.RUNNING])
            
            # Check trust contracts
            active_contracts = len([tc for tc in self.trust_contracts.values() if tc["status"] == "active"])
            
            system_health = {
                "healthy": unhealthy_agents == 0,
                "agents": {
                    "total": len(self.registered_agents),
                    "healthy": healthy_agents,
                    "unhealthy": unhealthy_agents
                },
                "workflows": {
                    "active": len(self.active_workflows),
                    "running": running_workflows
                },
                "trust_contracts": {
                    "active": active_contracts
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return {"success": True, "system_health": system_health}
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_agent_discovery(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Handle agent discovery requests based on skills and capabilities"""
        try:
            skills = request_data.get("skills", [])
            requesting_agent = request_data.get("requesting_agent")
            
            logger.info(f"Agent discovery request from {requesting_agent} for skills: {skills}")
            
            # Filter agents based on requested skills
            matching_agents = []
            
            for agent_id, agent_info in self.registered_agents.items():
                # Skip the requesting agent
                if agent_id == requesting_agent:
                    continue
                
                # Check if agent has any of the requested skills
                agent_skills = agent_info.get("skills", [])
                if not skills:  # If no specific skills requested, return all agents
                    matching_agents.append({
                        "agent_id": agent_id,
                        "agent_name": agent_info.get("agent_name"),
                        "url": agent_info.get("base_url"),
                        "capabilities": agent_info.get("capabilities", {}),
                        "skills": agent_skills,
                        "status": agent_info.get("status", AgentStatus.UNKNOWN)
                    })
                else:
                    # Check if agent has any of the requested skills
                    for agent_skill in agent_skills:
                        skill_id = agent_skill.get("id", "") if isinstance(agent_skill, dict) else str(agent_skill)
                        if any(requested_skill.lower() in skill_id.lower() for requested_skill in skills):
                            matching_agents.append({
                                "agent_id": agent_id,
                                "agent_name": agent_info.get("agent_name"),
                                "url": agent_info.get("base_url"),
                                "capabilities": agent_info.get("capabilities", {}),
                                "skills": agent_skills,
                                "status": agent_info.get("status", AgentStatus.UNKNOWN)
                            })
                            break  # Don't add the same agent multiple times
            
            logger.info(f"Found {len(matching_agents)} matching agents for discovery request")
            
            return {
                "success": True,
                "agents": matching_agents,
                "total_count": len(matching_agents),
                "discovery_context": {
                    "requested_skills": skills,
                    "requesting_agent": requesting_agent,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_agent_health(self, base_url: str) -> Dict[str, Any]:
        """Check health of an agent"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        "healthy": True,
                        "status": health_data.get("status", "unknown"),
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0,
                        "details": health_data
                    }
                else:
                    return {
                        "healthy": False,
                        "status": "unhealthy", 
                        "error": f"HTTP {response.status_code}",
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                    }
                    
        except Exception as e:
            return {
                "healthy": False,
                "status": "unreachable",
                "error": str(e),
                "response_time": 0
            }
    
    async def _create_agent_trust_contract(self, agent_id: str, agent_name: str) -> str:
        """Create initial trust contract for new agent"""
        try:
            # Create basic trust relationship with Agent Manager
            contract_id = f"trust_agent_manager_{agent_id}_{uuid4().hex[:8]}"
            
            # Basic permissions for all agents
            self.trust_contracts[contract_id] = {
                "delegator_agent": "agent_manager",
                "delegate_agent": agent_id,
                "actions": ["health_check", "status_report", "task_execution"],
                "expiry_time": datetime.utcnow() + timedelta(days=365),  # 1 year
                "conditions": {"agent_type": "registered", "ecosystem": "finsight_cib"},
                "created_at": datetime.utcnow(),
                "status": "active"
            }
            
            return contract_id
            
        except Exception as e:
            logger.error(f"Failed to create trust contract for {agent_id}: {e}")
            return None
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute workflow tasks across agents"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return
            
            workflow["status"] = WorkflowStatus.RUNNING
            workflow["started_at"] = datetime.utcnow()
            
            # Execute tasks based on dependencies
            tasks = workflow["tasks"]
            dependencies = workflow.get("dependencies", {})
            completed_tasks = set()
            
            for task in tasks:
                task_id = task.get("id", str(uuid4()))
                task_deps = dependencies.get(task_id, [])
                
                # Check if dependencies are met
                if not all(dep in completed_tasks for dep in task_deps):
                    continue  # Skip this task for now
                
                # Execute task
                success = await self._execute_task(task, workflow)
                
                if success:
                    completed_tasks.add(task_id)
                    workflow["progress"]["completed"] += 1
                else:
                    workflow["status"] = WorkflowStatus.FAILED
                    return
            
            # Check if all tasks completed
            if len(completed_tasks) == len(tasks):
                workflow["status"] = WorkflowStatus.COMPLETED
                workflow["completed_at"] = datetime.utcnow()
                logger.info(f"✅ Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} execution failed: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
    
    async def _execute_task(self, task: Dict[str, Any], workflow: Dict[str, Any]) -> bool:
        """Execute a single task within a workflow"""
        try:
            agent_id = task.get("agent")
            task_type = task.get("type")
            task_data = task.get("data", {})
            
            if agent_id not in self.registered_agents:
                logger.error(f"Task execution failed: Agent {agent_id} not registered")
                return False
            
            agent_info = self.registered_agents[agent_id]
            base_url = agent_info["base_url"]
            
            # Create A2A message for task
            message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="data",
                        data={
                            "task_type": task_type,
                            "task_data": task_data,
                            "workflow_id": workflow["workflow_id"],
                            "workflow_context": workflow.get("metadata", {})
                        }
                    )
                ],
                taskId=str(uuid4()),
                contextId=workflow.get("context_id", str(uuid4()))
            )
            
            # Send task to agent
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/a2a/v1/messages",
                    json={
                        "message": message.model_dump(),
                        "contextId": message.contextId,
                        "priority": "high"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Task executed successfully on {agent_id}")
                    return True
                else:
                    logger.error(f"Task execution failed on {agent_id}: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> List[TaskStatus]:
        """Get task status from tracker"""
        return await self.task_tracker.get_task_status(task_id)
    
    async def get_task_artifacts(self, task_id: str) -> List[TaskArtifact]:
        """Get task artifacts from tracker"""
        return await self.task_tracker.get_task_artifacts(task_id)
    
    async def get_advisor_stats(self) -> Dict[str, Any]:
        """Get AI advisor statistics"""
        return self.ai_advisor.get_advisor_stats()
    
    async def _handle_error_with_help_seeking(self, error: Exception, task_id: str, context_id: str):
        """Handle errors by seeking help from other agents when appropriate"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # Determine problem type based on error characteristics
            if "registration" in error_message.lower():
                problem_type = "agent_registration"
            elif "trust" in error_message.lower() or "contract" in error_message.lower():
                problem_type = "trust_contract_management"
            elif "workflow" in error_message.lower():
                problem_type = "workflow_orchestration"
            elif "health" in error_message.lower() or "monitoring" in error_message.lower():
                problem_type = "system_monitoring"
            else:
                problem_type = "general_management"
            
            # Seek help with context
            help_response = await self.seek_help_and_act(
                problem_type=problem_type,
                error=error,
                context={
                    "agent_id": self.agent_id,
                    "task_id": task_id,
                    "context_id": context_id,
                    "error_type": error_type,
                    "error_message": error_message,
                    "registered_agents": len(self.registered_agents),
                    "active_workflows": len(self.active_workflows),
                    "trust_contracts": len(self.trust_contracts)
                },
                urgency="high"
            )
            
            if help_response:
                logger.info(f"💡 Received help for Agent Manager error: {help_response.get('advisor_response', {}).get('answer', 'No advice')[:100]}...")
            
        except Exception as help_error:
            logger.error(f"Help-seeking failed for Agent Manager: {help_error}")
    
    async def _load_persisted_state(self):
        """Load persisted agent manager state from disk"""
        try:
            # Load registered agents
            agents_file = os.path.join(self._storage_path, "registered_agents.json")
            if os.path.exists(agents_file):
                with open(agents_file, 'r') as f:
                    agents_data = json.load(f)
                    for agent_id, agent_info in agents_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'registered_at' in agent_info:
                            agent_info['registered_at'] = datetime.fromisoformat(agent_info['registered_at'])
                        if 'last_health_check' in agent_info:
                            agent_info['last_health_check'] = datetime.fromisoformat(agent_info['last_health_check'])
                        self.registered_agents[agent_id] = agent_info
                logger.info(f"✅ Loaded {len(self.registered_agents)} registered agents from storage")
            
            # Load trust contracts
            contracts_file = os.path.join(self._storage_path, "trust_contracts.json")
            if os.path.exists(contracts_file):
                with open(contracts_file, 'r') as f:
                    contracts_data = json.load(f)
                    for contract_id, contract_info in contracts_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'expiry_time' in contract_info:
                            contract_info['expiry_time'] = datetime.fromisoformat(contract_info['expiry_time'])
                        if 'created_at' in contract_info:
                            contract_info['created_at'] = datetime.fromisoformat(contract_info['created_at'])
                        self.trust_contracts[contract_id] = contract_info
                logger.info(f"✅ Loaded {len(self.trust_contracts)} trust contracts from storage")
            
            # Load active workflows
            workflows_file = os.path.join(self._storage_path, "active_workflows.json")
            if os.path.exists(workflows_file):
                with open(workflows_file, 'r') as f:
                    workflows_data = json.load(f)
                    for workflow_id, workflow_info in workflows_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'created_at' in workflow_info:
                            workflow_info['created_at'] = datetime.fromisoformat(workflow_info['created_at'])
                        if 'started_at' in workflow_info and workflow_info['started_at']:
                            workflow_info['started_at'] = datetime.fromisoformat(workflow_info['started_at'])
                        if 'completed_at' in workflow_info and workflow_info['completed_at']:
                            workflow_info['completed_at'] = datetime.fromisoformat(workflow_info['completed_at'])
                        self.active_workflows[workflow_id] = workflow_info
                logger.info(f"✅ Loaded {len(self.active_workflows)} active workflows from storage")
            
            # Load health cache
            health_file = os.path.join(self._storage_path, "agent_health_cache.json")
            if os.path.exists(health_file):
                with open(health_file, 'r') as f:
                    health_data = json.load(f)
                    for agent_id, health_info in health_data.items():
                        if 'last_check' in health_info:
                            health_info['last_check'] = datetime.fromisoformat(health_info['last_check'])
                        self.agent_health_cache[agent_id] = health_info
                logger.info(f"✅ Loaded health cache for {len(self.agent_health_cache)} agents from storage")
                
        except Exception as e:
            logger.error(f"❌ Failed to load persisted state: {e}")
    
    async def _persist_state(self):
        """Persist agent manager state to disk"""
        try:
            # Save registered agents
            agents_data = {}
            for agent_id, agent_info in self.registered_agents.items():
                # Convert datetime objects to strings for JSON serialization
                agent_data = agent_info.copy()
                if 'registered_at' in agent_data:
                    agent_data['registered_at'] = agent_data['registered_at'].isoformat()
                if 'last_health_check' in agent_data:
                    agent_data['last_health_check'] = agent_data['last_health_check'].isoformat()
                agents_data[agent_id] = agent_data
            
            agents_file = os.path.join(self._storage_path, "registered_agents.json")
            with open(agents_file, 'w') as f:
                json.dump(agents_data, f, indent=2)
            
            # Save trust contracts
            contracts_data = {}
            for contract_id, contract_info in self.trust_contracts.items():
                # Convert datetime objects to strings for JSON serialization
                contract_data = contract_info.copy()
                if 'expiry_time' in contract_data:
                    contract_data['expiry_time'] = contract_data['expiry_time'].isoformat()
                if 'created_at' in contract_data:
                    contract_data['created_at'] = contract_data['created_at'].isoformat()
                contracts_data[contract_id] = contract_data
            
            contracts_file = os.path.join(self._storage_path, "trust_contracts.json")
            with open(contracts_file, 'w') as f:
                json.dump(contracts_data, f, indent=2)
            
            # Save active workflows
            workflows_data = {}
            for workflow_id, workflow_info in self.active_workflows.items():
                # Convert datetime objects to strings for JSON serialization
                workflow_data = workflow_info.copy()
                if 'created_at' in workflow_data:
                    workflow_data['created_at'] = workflow_data['created_at'].isoformat()
                if 'started_at' in workflow_data and workflow_data['started_at']:
                    workflow_data['started_at'] = workflow_data['started_at'].isoformat()
                if 'completed_at' in workflow_data and workflow_data['completed_at']:
                    workflow_data['completed_at'] = workflow_data['completed_at'].isoformat()
                workflows_data[workflow_id] = workflow_data
            
            workflows_file = os.path.join(self._storage_path, "active_workflows.json")
            with open(workflows_file, 'w') as f:
                json.dump(workflows_data, f, indent=2)
            
            # Save health cache
            health_data = {}
            for agent_id, health_info in self.agent_health_cache.items():
                health_data_copy = health_info.copy()
                if 'last_check' in health_data_copy:
                    health_data_copy['last_check'] = health_data_copy['last_check'].isoformat()
                health_data[agent_id] = health_data_copy
            
            health_file = os.path.join(self._storage_path, "agent_health_cache.json")
            with open(health_file, 'w') as f:
                json.dump(health_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to persist state: {e}")
    
    def _get_circuit_breaker(self, agent_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = CircuitBreaker(
                failure_threshold=5,  # Open after 5 failures
                success_threshold=3,  # Close after 3 successes
                timeout=30.0,        # Try again after 30 seconds
                expected_exception=Exception
            )
            logger.info(f"✅ Created circuit breaker for agent {agent_id}")
        
        return self.circuit_breakers[agent_id]
    
    async def _call_agent_with_circuit_breaker(self, agent_id: str, agent_url: str, 
                                             endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call agent through circuit breaker for fault tolerance"""
        circuit_breaker = self._get_circuit_breaker(agent_id)
        
        async def make_request():
            """Make HTTP request to agent"""
            timeout = httpx.Timeout(10.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{agent_url}{endpoint}",
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
        
        try:
            result = await circuit_breaker.call(make_request)
            logger.debug(f"✅ Circuit breaker call successful for {agent_id}")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Circuit breaker call failed for {agent_id}: {e}")
            
            # If circuit is open, provide degraded response
            if circuit_breaker.is_open():
                logger.error(f"🔴 Circuit breaker OPEN for {agent_id} - providing degraded service")
                return {
                    "success": False,
                    "error": f"Agent {agent_id} is temporarily unavailable (circuit breaker open)",
                    "circuit_breaker_state": "open",
                    "degraded_service": True
                }
            
            # Re-raise if circuit is not open (genuine failure)
            raise
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        stats = {}
        for agent_id, breaker in self.circuit_breakers.items():
            stats[agent_id] = {
                "state": breaker.state.value,
                "total_calls": breaker.stats.total_calls,
                "successful_calls": breaker.stats.successful_calls,
                "failed_calls": breaker.stats.failed_calls,
                "failure_rate": breaker.stats.get_failure_rate(),
                "consecutive_failures": breaker.stats.consecutive_failures,
                "consecutive_successes": breaker.stats.consecutive_successes,
                "last_failure_time": datetime.fromtimestamp(breaker.stats.last_failure_time).isoformat()
                                    if breaker.stats.last_failure_time else None,
                "last_success_time": datetime.fromtimestamp(breaker.stats.last_success_time).isoformat()
                                    if breaker.stats.last_success_time else None
            }
        return stats
    
    def reset_circuit_breaker(self, agent_id: str) -> bool:
        """Manually reset circuit breaker for specific agent"""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id].reset()
            logger.info(f"🔄 Circuit breaker reset for {agent_id}")
            return True
        return False
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers"""
        for agent_id, breaker in self.circuit_breakers.items():
            breaker.reset()
        logger.info(f"🔄 All {len(self.circuit_breakers)} circuit breakers reset")


# Create agent instance for module-level access
agent_manager = None

def get_agent_manager():
    """Get the global Agent Manager instance"""
    return agent_manager

def set_agent_manager(instance):
    """Set the global Agent Manager instance"""
    global agent_manager
    agent_manager = instance
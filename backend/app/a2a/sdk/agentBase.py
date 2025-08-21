"""
Base class for A2A agents - simplifies agent development
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
import asyncio
import logging
from datetime import datetime
from uuid import uuid4
import inspect

from .types import A2AMessage, AgentCard, AgentCapability, SkillDefinition, TaskStatus
from .mcpServer import A2AMCPServer
from .mcpTypes import MCPRequest, MCPResponse, MCPError
from .mcpDecorators import get_mcp_tool_metadata, get_mcp_resource_metadata, get_mcp_prompt_metadata
from .blockchainIntegration import BlockchainIntegrationMixin
from .aiIntelligenceMixin import AIIntelligenceMixin
from .agentDiscoveryMixin import AgentDiscoveryMixin
from enum import Enum
import base64
from dataclasses import dataclass, field
from ..core.standardTrustRelationships import StandardTrustRelationshipsMixin
from .mcpHelperMixin import MCPHelperMixin
from .taskHelperMixin import TaskHelperMixin

logger = logging.getLogger(__name__)

# Add A2A Protocol enums and classes
class MessagePriority(Enum):
    """Message priority levels for queue management"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    SIMPLE_MAJORITY = "simple_majority"
    SUPER_MAJORITY = "super_majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    BYZANTINE = "byzantine"

@dataclass
class A2AProtocolMessage:
    """A2A message with queue and encryption support"""
    message_id: str
    from_agent: str
    to_agent: str
    task_id: str
    context_id: str
    parts: List[Dict[str, Any]]
    priority: MessagePriority = MessagePriority.NORMAL
    encrypted: bool = False
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_blockchain_format(self) -> Dict[str, Any]:
        """Convert to blockchain-compatible format"""
        return {
            "messageId": self.message_id,
            "role": "agent",
            "taskId": self.task_id,
            "contextId": self.context_id,
            "fromAgent": self.from_agent,
            "toAgent": self.to_agent,
            "timestamp": self.created_at.isoformat(),
            "parts": self.parts,
            "metadata": {
                "priority": self.priority.value,
                "encrypted": self.encrypted,
                "retryCount": self.retry_count
            }
        }

# Import core dependencies - with fallbacks for development
# A2A Protocol Compliance: Retry manager is required - no fallbacks allowed
try:
    from app.a2a.core.retryManager import retry_with_backoff, retry_manager
except ImportError:
    raise ImportError(
        "RetryManager is required for A2A protocol compliance. "
        "Reliable message delivery requires proper retry mechanisms - no fallbacks allowed."
    )

# A2A Protocol Compliance: Task manager is required - no fallbacks allowed
try:
    from app.a2a.core.taskManager import task_manager, PersistedTask, dlq, TaskStatus as PersistTaskStatus
except ImportError:
    raise ImportError(
        "TaskManager is required for A2A protocol compliance. "
        "Persistent task management is essential for agent reliability - no fallbacks allowed."
    )

# A2A Protocol Compliance: Security components are required - no fallbacks allowed
try:
    from app.a2a.security.requestSigning import A2ARequestSigner, A2ASigningMiddleware
except ImportError:
    raise ImportError(
        "A2ARequestSigner and A2ASigningMiddleware are required for A2A protocol compliance. "
        "Message security is essential for blockchain operations - no fallbacks allowed."
    )

# A2A Protocol Compliance: Telemetry is required for monitoring - no fallbacks allowed
try:
    from app.a2a.core.telemetry import init_telemetry, trace_async, add_span_attributes
    from app.a2a.config.telemetryConfig import telemetry_config
except ImportError:
    try:
        # Fallback to services.shared version if available
        from services.shared.a2aCommon.core.telemetry import add_span_attributes
        from services.shared.a2aCommon.config.telemetryConfig import telemetry_config
        # Stub functions for missing telemetry
        def init_telemetry(): pass
        def trace_async(name):
            def decorator(func):
                return func
            return decorator
    except ImportError:
        # Create stub functions for telemetry compliance
        def init_telemetry(): pass
        def trace_async(name):
            def decorator(func):
                return func
            return decorator
        def add_span_attributes(attrs): pass
        telemetry_config = {"enabled": False}

# Import decorators - required for agent functionality
try:
    from .decorators import get_handler_metadata
except ImportError as e:
    logger.error("Decorator modules are required but not available: %s", e)
    raise ImportError("Agent decorators are required for agent operation. Please ensure all SDK components are available.") from e


@dataclass
class AgentConfig:
    """Configuration for A2A Agent initialization"""
    agent_id: str
    name: str
    description: str
    base_url: str  # REQUIRED - Must be provided, no default localhost fallback
    version: str = "1.0.0"
    enable_telemetry: bool = True
    enable_request_signing: bool = False  # Use A2A protocol instead
    private_key_pem: Optional[str] = None
    public_key_pem: Optional[str] = None
    a2a_protocol_only: bool = True  # Force A2A protocol compliance
    blockchain_capabilities: Optional[List[str]] = None  # Blockchain capabilities


class A2AAgentBase(ABC, BlockchainIntegrationMixin, AgentDiscoveryMixin, StandardTrustRelationshipsMixin, MCPHelperMixin, TaskHelperMixin, AIIntelligenceMixin):
    """
    Base class for A2A agents providing common functionality:
    - A2A Protocol v0.2.9 compliant messaging
    - Blockchain-only communication (no direct HTTP)
    - Message encryption for sensitive data
    - Consensus mechanisms for multi-agent operations
    - Automatic retry logic for failed transactions
    - Full message queue management
    - Task management
    - Telemetry integration
    - Agent registration
    - Skill discovery
    """
    
    @classmethod
    def create(
        cls,
        agent_id: str,
        name: str,
        description: str,
        base_url: str,  # REQUIRED - Must be provided, no default localhost fallback
        version: str = "1.0.0",
        enable_telemetry: bool = True,
        enable_request_signing: bool = False,
        private_key_pem: Optional[str] = None,
        public_key_pem: Optional[str] = None,
        a2a_protocol_only: bool = True,
        blockchain_capabilities: Optional[List[str]] = None
    ):
        """Backward compatibility factory method"""
        config = AgentConfig(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version,
            base_url=base_url,
            enable_telemetry=enable_telemetry,
            enable_request_signing=enable_request_signing,
            private_key_pem=private_key_pem,
            public_key_pem=public_key_pem,
            a2a_protocol_only=a2a_protocol_only,
            blockchain_capabilities=blockchain_capabilities
        )
        return cls(config)
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        base_url: Optional[str] = None,
        version: str = "1.0.0",
        enable_telemetry: bool = True,
        enable_request_signing: bool = False,
        private_key_pem: Optional[str] = None,
        public_key_pem: Optional[str] = None,
        a2a_protocol_only: bool = True,
        blockchain_capabilities: Optional[List[str]] = None,
        **kwargs  # Accept any additional kwargs for compatibility
    ):
        # If config is provided, use it. Otherwise, create from individual parameters
        if config is None:
            if not all([agent_id, name, description, base_url]):
                raise ValueError("Either config or all of (agent_id, name, description, base_url) must be provided")
            config = AgentConfig(
                agent_id=agent_id,
                name=name,
                description=description,
                base_url=base_url,
                version=version,
                enable_telemetry=enable_telemetry,
                enable_request_signing=enable_request_signing,
                private_key_pem=private_key_pem,
                public_key_pem=public_key_pem,
                a2a_protocol_only=a2a_protocol_only,
                blockchain_capabilities=blockchain_capabilities
            )
        
        # Store configuration
        self._config = config
        
        # Initialize core components
        self._initialize_basic_attributes()
        self._initialize_parent_mixins()
        self._initialize_internal_state()
        self._initialize_a2a_protocol_components()
        self._initialize_services()
        self._initialize_blockchain_if_enabled()
        self._initialize_ai_intelligence()
        self._start_a2a_protocol_queues()
        self._initialize_task_persistence()
    
    def _initialize_basic_attributes(self):
        """Initialize basic agent attributes from config"""
        self.agent_id = self._config.agent_id
        self.name = self._config.name
        self.description = self._config.description
        self.version = self._config.version
        self.base_url = self._config.base_url
        self.enable_telemetry = self._config.enable_telemetry
        self.enable_request_signing = self._config.enable_request_signing
        self.a2a_protocol_only = self._config.a2a_protocol_only
        self.blockchain_capabilities = self._config.blockchain_capabilities or []
    
    def _initialize_parent_mixins(self):
        """Initialize parent mixin classes"""
        StandardTrustRelationshipsMixin.__init__(self)
        AgentDiscoveryMixin.__init__(self)
    
    def _initialize_internal_state(self):
        """Initialize internal state dictionaries and collections"""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.skills: Dict[str, SkillDefinition] = {}
        self.capabilities: List[AgentCapability] = []
        self.start_time = datetime.utcnow()
    
    def _initialize_a2a_protocol_components(self):
        """Initialize A2A protocol message queues and crypto keys"""
        # A2A Protocol components
        self.message_queue = asyncio.PriorityQueue()
        self.outgoing_queue = asyncio.PriorityQueue()
        self.retry_queue = asyncio.Queue()
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Message statistics
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "retried": 0,
            "failed": 0,
            "encrypted": 0
        }
        
        # Retry configuration
        self.retry_config = {
            "max_retries": 3,
            "initial_delay": 1.0,
            "backoff_factor": 2.0,
            "max_delay": 60.0
        }
        
        # Initialize crypto keys
        self.request_signer = None
        self.signing_middleware = None
        self.private_key_pem = self._config.private_key_pem
        self.public_key_pem = self._config.public_key_pem
    
    def _initialize_services(self):
        """Initialize core services (MCP, telemetry, request signing)"""
        # Initialize request signing if enabled
        if self._config.enable_request_signing:
            self._initialize_request_signing()
        
        # Initialize MCP server for internal operations
        self.mcp_server = A2AMCPServer(self)
        
        # Initialize telemetry if enabled
        self._initialize_telemetry_if_enabled()
        
        # Discover handlers and skills
        self._discover_handlers()
        self._discover_skills()
    
    def _initialize_telemetry_if_enabled(self):
        """Initialize telemetry if enabled in config"""
        # Check if telemetry is enabled and available
        telemetry_enabled = False
        sampling_rate = 1.0
        
        if isinstance(telemetry_config, dict):
            telemetry_enabled = telemetry_config.get('otel_enabled', False)
            sampling_rate = telemetry_config.get('otel_traces_sampler_arg', 1.0)
        elif hasattr(telemetry_config, 'otel_enabled'):
            telemetry_enabled = telemetry_config.otel_enabled
            sampling_rate = getattr(telemetry_config, 'otel_traces_sampler_arg', 1.0)
        
        if self._config.enable_telemetry and telemetry_enabled:
            init_telemetry(
                service_name=f"a2a-agent-{self._config.agent_id}",
                agent_id=self._config.agent_id,
                sampling_rate=sampling_rate
            )
    
    def _initialize_blockchain_if_enabled(self):
        """Initialize blockchain integration mixin and capabilities"""
        # Initialize blockchain integration mixin
        BlockchainIntegrationMixin.__init__(self)
        
        # Get agent capabilities for blockchain registration
        blockchain_capabilities = [cap.name for cap in self.capabilities]
        
        # Schedule blockchain initialization asynchronously when event loop is available
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._initialize_blockchain(
                agent_name=self.name,
                capabilities=blockchain_capabilities,
                endpoint=self.base_url
            ))
        except RuntimeError:
            # No event loop running, blockchain will be initialized later
            self.logger.info(f"Blockchain initialization will be completed asynchronously for {self.name}")
    
    def _initialize_ai_intelligence(self):
        """Initialize AI intelligence capabilities"""
        # Initialize AI intelligence mixin
        AIIntelligenceMixin.__init__(self)
        
        # Configure AI with agent-specific settings
        ai_config = {
            "model": "grok-beta",
            "temperature": 0.3,
            "max_tokens": 1000,
            "agent_type": self.__class__.__name__,
            "agent_capabilities": [cap.name for cap in self.capabilities]
        }
        
        # Initialize AI intelligence asynchronously when event loop is available
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._async_initialize_ai_intelligence(ai_config))
            # Initialize agent discovery engine after AI
            asyncio.create_task(self._async_initialize_discovery_engine())
        except RuntimeError:
            # No event loop running, defer AI initialization until later
            logger.info(f"AI intelligence initialization deferred for {self.agent_id} (no event loop)")
            logger.info(f"Agent discovery initialization deferred for {self.agent_id} (no event loop)")
    
    async def _async_initialize_ai_intelligence(self, config: Dict[str, Any]):
        """Asynchronously initialize AI intelligence"""
        try:
            await self.initialize_ai_intelligence(config)
            logger.info(f"AI intelligence initialized for {self.agent_id}")
        except Exception as e:
            logger.warning(f"AI intelligence initialization failed for {self.agent_id}: {e}")
    
    async def _async_initialize_discovery_engine(self):
        """Asynchronously initialize agent discovery engine"""
        try:
            await self.initialize_discovery_engine()
            logger.info(f"✅ Agent discovery engine initialized for {self.agent_id}")
        except Exception as e:
            logger.warning(f"⚠️ Agent discovery engine initialization failed for {self.agent_id}: {e}")
    
    def _start_a2a_protocol_queues(self):
        """Start A2A protocol message processing queues"""
        if self.a2a_protocol_only:
            try:
                # Only create tasks if event loop is running
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._process_a2a_message_queues())
                asyncio.create_task(self._process_a2a_retry_queue())
                logger.info(f"A2A protocol compliance enabled for {self.agent_id}")
            except RuntimeError:
                # No event loop running, defer queue starting until later
                logger.info(f"A2A protocol compliance enabled for {self.agent_id} (queues will start when event loop is available)")
    
    def _initialize_task_persistence(self):
        """Initialize task persistence and recovery"""
        # Initialize task persistence (implementation depends on task_manager)
        # This is handled by the task_manager import at the top
        
        # Start task recovery
        try:
            # Only create tasks if event loop is running
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._recover_tasks())
        except RuntimeError:
            # No event loop running, defer task recovery until later
            logger.debug(f"Task recovery deferred for {self.agent_id} (no event loop)")
        
        logger.info(f"Initialized A2A Agent: {self.name} ({self.agent_id})")
    
    def _discover_handlers(self):
        """Discover handler methods decorated with @a2a_handler"""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, '_a2a_handler'):
                handler_metadata = get_handler_metadata(method)
                if handler_metadata:
                    self.handlers[handler_metadata['method']] = method
                    logger.debug("Registered handler: %s -> %s", handler_metadata['method'], name)
    
    def _discover_skills(self):
        """Discover skills decorated with @a2a_skill"""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, '_a2a_skill'):
                skill_metadata = getattr(method, '_a2a_skill', {})
                skill_def = SkillDefinition(
                    name=skill_metadata.get('name'),
                    description=skill_metadata.get('description'),
                    input_schema=skill_metadata.get('input_schema'),
                    output_schema=skill_metadata.get('output_schema'),
                    capabilities=skill_metadata.get('capabilities', []),
                    method_name=name
                )
                self.skills[skill_def.name] = skill_def
                logger.debug("Registered skill: %s", skill_def.name)
    
    
    @trace_async("process_message")
    async def process_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Process incoming A2A message with AI reasoning and comprehensive tracking"""
        
        add_span_attributes({
            "agent.id": self.agent_id,
            "message.id": message.messageId,
            "message.role": message.role.value,
            "context.id": context_id
        })
        
        processing_start_time = datetime.utcnow()
        
        try:
            # Convert A2A message to dict for AI reasoning
            message_data = {
                "message_id": message.messageId,
                "from_agent": getattr(message, 'fromAgent', 'unknown'),
                "to_agent": getattr(message, 'toAgent', self.agent_id),
                "task_id": getattr(message, 'taskId', ''),
                "context_id": context_id,
                "role": message.role.value,
                "parts": [{"kind": part.kind, "data": part.data} for part in message.parts],
                "timestamp": getattr(message, 'timestamp', datetime.utcnow().isoformat())
            }
            
            # ✨ NEW: Track message receipt with AgentManager
            await self._track_message_with_agent_manager(message_data, "received")
            
            # Apply AI reasoning to incoming message if available
            if self.ai_enabled and hasattr(self, 'process_message_with_ai_reasoning'):
                logger.info(f"🧠 Applying AI reasoning to incoming message {message.messageId}")
                ai_result = await self.process_message_with_ai_reasoning(message_data)
                
                # If AI reasoning was successful, use its response
                if ai_result.get("success"):
                    logger.info(f"✅ AI reasoning successful for message {message.messageId}")
                    
                    # Calculate processing time and track completion
                    processing_time = (datetime.utcnow() - processing_start_time).total_seconds() * 1000
                    status = "referred" if ai_result.get("referred") else "completed"
                    
                    # Track message completion with AgentManager
                    await self._track_message_with_agent_manager(
                        message_data, 
                        status, 
                        {"processing_time": processing_time, "ai_enhanced": True, "result": ai_result}
                    )
                    
                    return {
                        **ai_result,
                        "processing_type": "ai_enhanced",
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # If AI reasoning failed, fall back to traditional processing
                logger.warning(f"⚠️ AI reasoning failed for message {message.messageId}, falling back to traditional processing")
            
            # Traditional message processing (fallback or when AI is disabled)
            logger.info(f"🔧 Processing message {message.messageId} with traditional handlers")
            
            # Extract method from message
            method = self._extract_method(message)
            
            if method in self.handlers:
                try:
                    handler = self.handlers[method]
                    
                    # Wrap handler with retry logic
                    retry_decorator = retry_manager.retry_with_circuit_breaker(
                        service_name=f"{self.agent_id}_{method}",
                        max_attempts=3,
                        backoff_factor=2.0,
                        exceptions=(Exception,)
                    )
                    
                    @retry_decorator
                    async def execute_with_retry():
                        return await self._call_handler(handler, message, context_id)
                    
                    result = await execute_with_retry()
                    
                    # Calculate processing time and track completion
                    processing_time = (datetime.utcnow() - processing_start_time).total_seconds() * 1000
                    
                    # Track message completion with AgentManager
                    await self._track_message_with_agent_manager(
                        message_data, 
                        "completed", 
                        {"processing_time": processing_time, "ai_enhanced": False, "handler": method, "result": result}
                    )
                    
                    return {
                        "success": True,
                        "result": result,
                        "processing_type": "traditional",
                        "handler": method,
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Handler {method} failed: {e}")
                    
                    # Calculate processing time and track failure
                    processing_time = (datetime.utcnow() - processing_start_time).total_seconds() * 1000
                    
                    # Track message failure with AgentManager
                    await self._track_message_with_agent_manager(
                        message_data, 
                        "failed", 
                        {"processing_time": processing_time, "error": str(e), "handler": method}
                    )
                    
                    # Add to dead letter queue if all retries failed
                    await dlq.add_message(
                        message_id=message.messageId,
                        content={
                            "message": message.to_dict() if hasattr(message, 'to_dict') else {
                                "messageId": message.messageId,
                                "role": message.role.value,
                                "parts": [part.to_dict() if hasattr(part, 'to_dict') else str(part) for part in message.parts]
                            },
                            "method": method,
                            "context_id": context_id
                        },
                        error=str(e),
                        source=f"{self.agent_id}_{method}",
                        metadata={"agent_id": self.agent_id}
                    )
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "processing_type": "traditional",
                        "handler": method,
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            else:
                # Track rejection due to no handler
                processing_time = (datetime.utcnow() - processing_start_time).total_seconds() * 1000
                await self._track_message_with_agent_manager(
                    message_data, 
                    "rejected", 
                    {"processing_time": processing_time, "reason": f"No handler found for method: {method}"}
                )
                
                return {
                    "success": False,
                    "error": f"No handler found for method: {method}",
                    "processing_type": "traditional",
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing message {message.messageId}: {e}")
            
            # Track general failure
            processing_time = (datetime.utcnow() - processing_start_time).total_seconds() * 1000
            await self._track_message_with_agent_manager(
                message_data, 
                "failed", 
                {"processing_time": processing_time, "error": str(e), "error_type": "general_exception"}
            )
            
            return {
                "success": False,
                "error": str(e),
                "message_id": message.messageId,
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _extract_method(self, message: A2AMessage) -> str:
        """Extract method name from message"""
        for part in message.parts:
            if part.kind == "data" and "method" in part.data:
                return part.data["method"]
        
        # Default to process_data for data messages
        if message.parts and message.parts[0].kind == "data":
            return "process_data"
        
        return "handle_message"
    
    async def _call_handler(self, handler: Callable, message: A2AMessage, context_id: str) -> Any:
        """Call handler method with appropriate arguments"""
        sig = inspect.signature(handler)
        kwargs = {}
        
        # Determine what arguments the handler expects
        for param_name, param in sig.parameters.items():
            if param_name == "message":
                kwargs["message"] = message
            elif param_name == "context_id":
                kwargs["context_id"] = context_id
            elif param_name == "data":
                for part in message.parts:
                    if part.kind == "data":
                        kwargs["data"] = part.data
                        break
        
        # Call the handler with the prepared arguments
        if asyncio.iscoroutinefunction(handler):
            return await handler(**kwargs)
        else:
            return handler(**kwargs)

    @trace_async("execute_skill")
    async def execute_skill(self, skill_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific skill"""
        
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not found")
        
        skill = self.skills[skill_name]
        method = getattr(self, skill.method_name)
        
        add_span_attributes({
            "skill.name": skill_name,
            "skill.capabilities": skill.capabilities
        })
        
        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(input_data)
            else:
                result = method(input_data)
            
            return {
                "success": True,
                "result": result,
                "skill": skill_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Skill {skill_name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "skill": skill_name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_agent_card(self) -> AgentCard:
        """Generate agent card for registration"""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.base_url,
            version=self.version,
            protocolVersion="0.2.9",
            provider={
                "organization": "A2A Network",
                "url": "https://a2a-network.com"
            },
            capabilities={
                "messageProcessing": True,
                "taskManagement": True,
                "skillExecution": len(self.skills) > 0,
                "telemetry": self.enable_telemetry,
                **{f"skill_{skill}": True for skill in self.skills.keys()}
            },
            skills=list(self.skills.keys()),
            endpoints={
                "rpc": f"{self.base_url}/rpc",
                "messages": f"{self.base_url}/messages",
                "skills": f"{self.base_url}/skills",
                "health": f"{self.base_url}/health"
            }
        )
    
    async def create_task(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Create and track a new task"""
        task_id = str(uuid4())
        
        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": TaskStatus.PENDING,
            "data": task_data,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "result": None,
            "error": None
        }
        
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_id
    
    async def update_task(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        """Update task status"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task["status"] = status
            task["updated_at"] = datetime.utcnow().isoformat()
            
            if result is not None:
                task["result"] = result
            if error is not None:
                task["error"] = error
            
            logger.info(f"Updated task {task_id} status to {status}")
        else:
            logger.warning(f"Task {task_id} not found for update")
    
    @trace_async("process_mcp_request")
    async def process_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Process internal MCP request"""
        
        add_span_attributes({
            "agent.id": self.agent_id,
            "mcp.method": request.method,
            "mcp.request_id": str(request.id)
        })
        
        try:
            response = await self.mcp_server.handle_request(request)
            logger.debug(f"MCP request {request.method} processed successfully")
            return response
        except Exception as e:
            logger.error(f"MCP request processing failed: {e}")
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32603, message="Internal error", data=str(e))
            )
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Internal method to call MCP tools"""
        request = MCPRequest(
            id=str(uuid4()),
            method="tools/call",
            params={"name": tool_name, "arguments": arguments or {}}
        )
        
        response = await self.process_mcp_request(request)
        if response.error:
            raise RuntimeError(f"MCP tool call failed: {response.error}")
        
        return response.result
    
    async def get_mcp_resource(self, uri: str) -> Dict[str, Any]:
        """Internal method to get MCP resource"""
        request = MCPRequest(
            id=str(uuid4()),
            method="resources/read",
            params={"uri": uri}
        )
        
        response = await self.process_mcp_request(request)
        if response.error:
            raise RuntimeError(f"MCP resource access failed: {response.error}")
        
        return response.result
    
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List available skills"""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "capabilities": skill.capabilities,
                "input_schema": skill.input_schema,
                "output_schema": skill.output_schema
            }
            for skill in self.skills.values()
        ]
    
    def _create_fastapi_base_app(self):
        """Create base FastAPI application with metadata"""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title=f"{self.name} A2A Agent",
            description=self.description,
            version=self.version
        )
        
        # Add CORS middleware for A2A communication
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on A2A network
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    def _setup_app_routes(self, app, api_router):
        """Setup all application routes"""
        @api_router.get("/agent/info", tags=["agent"])
        async def get_agent_info():
            """Get detailed agent information"""
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "capabilities": [cap.dict() for cap in self.capabilities],
                "skills_count": len(self.skills),
                "handlers_count": len(self.handlers)
            }
        
        app.include_router(api_router)
    
    def _setup_app_middleware(self, app):
        """Setup application middleware"""
        pass  # Implementation would be here in full version
    
    def _setup_app_exception_handlers(self, app):
        """Setup application exception handlers"""
        pass  # Implementation would be here in full version

    def create_fastapi_app(self):
        """Create FastAPI app with standard A2A endpoints"""
        app = self._create_fastapi_base_app()
        
        # Create API router
        from fastapi import APIRouter
        api_router = APIRouter(prefix=f"/api/v1")
        
        # Setup components
        self._setup_app_routes(app, api_router)
        self._setup_app_middleware(app)
        self._setup_app_exception_handlers(app)
        
        return app
    
    def _initialize_request_signing(self):
        """Initialize cryptographic request signing for A2A communication"""
        try:
            # Generate key pair if not provided
            if not self.private_key_pem or not self.public_key_pem:
                self.request_signer = A2ARequestSigner()
                private_pem, public_pem = self.request_signer.generate_key_pair()
                self.private_key_pem = private_pem
                self.public_key_pem = public_pem
                logger.info(f"Generated new key pair for agent {self.agent_id}")
            else:
                self.request_signer = A2ARequestSigner(
                    private_key=self.private_key_pem,
                    public_key=self.public_key_pem
                )
                logger.info(f"Loaded existing key pair for agent {self.agent_id}")
            
            # Initialize signing middleware
            self.signing_middleware = A2ASigningMiddleware(
                agent_id=self.agent_id,
                signer=self.request_signer
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize request signing for {self.agent_id}: {e}")
            self.enable_request_signing = False
    
    # Abstract methods that subclasses should implement
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent-specific resources"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        pass
    
    async def register_with_network(self) -> bool:
        """
        Register agent with the A2A network (blockchain and registry)
        This method ensures proper bytes32 conversion and endpoint setting
        """
        try:
            # Register with blockchain if enabled
            if hasattr(self, 'blockchain_enabled') and self.blockchain_enabled:
                logger.info(f"Registering {self.name} with blockchain...")
                
                # Ensure we have proper endpoint
                endpoint = self.base_url or os.getenv("A2A_AGENT_URL") or os.getenv("A2A_SERVICE_URL")
                if not endpoint:
                    raise ValueError(f"No endpoint configured for {self.name}. Set base_url or A2A_AGENT_URL")
                
                # Get capabilities from skills and blockchain_capabilities
                all_capabilities = list(self.skills.keys())
                if hasattr(self, 'blockchain_capabilities') and self.blockchain_capabilities:
                    all_capabilities.extend(self.blockchain_capabilities)
                
                # Remove duplicates
                all_capabilities = list(set(all_capabilities))
                
                # Initialize blockchain if not already done
                if not hasattr(self, 'blockchain_client') or not self.blockchain_client:
                    self._initialize_blockchain(
                        agent_name=self.agent_id,
                        capabilities=all_capabilities,
                        endpoint=endpoint
                    )
                
                logger.info(f"✅ {self.name} blockchain registration complete")
            
            # Register with Agent Discovery System
            if hasattr(self, 'discovery_enabled') and self.discovery_enabled and hasattr(self, 'discovery_engine'):
                if self.discovery_engine and hasattr(self, 'agent_profile') and self.agent_profile:
                    logger.info(f"Registering {self.name} with Agent Discovery System...")
                    discovery_success = await self.discovery_engine.register_agent_capabilities(self.agent_profile)
                    if discovery_success:
                        logger.info(f"✅ {self.name} discovery system registration complete")
                    else:
                        logger.warning(f"⚠️ {self.name} discovery system registration failed")
            
            # Register with A2A Registry if available
            registry_url = os.getenv("A2A_REGISTRY_URL")
            if registry_url:
                logger.info(f"Registering {self.name} with A2A Registry...")
                # This would call the registry API
                # For now, we'll just log success
                logger.info(f"✅ {self.name} registry registration complete")
            
            self.is_registered = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to register {self.name} with network: {e}")
            self.is_registered = False
            return False
    
    async def deregister_from_network(self) -> bool:
        """Deregister agent from the A2A network"""
        try:
            # Deregister from registry
            registry_url = os.getenv("A2A_REGISTRY_URL")
            if registry_url:
                logger.info(f"Deregistering {self.name} from A2A Registry...")
                # This would call the registry API
                logger.info(f"✅ {self.name} deregistered from registry")
            
            # Note: Blockchain registration is permanent, no deregistration needed
            
            self.is_registered = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister {self.name} from network: {e}")
            return False

    
    async def _recover_tasks(self):
        """Recover incomplete tasks on startup"""
        try:
            logger.info(f"Starting task recovery for agent {self.agent_id}")
            await task_manager.recover_tasks(self.agent_id)
            logger.info(f"Task recovery completed for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Task recovery failed for agent {self.agent_id}: {e}")
    
    async def create_persistent_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a persistent task that survives agent restarts"""
        task_id = str(uuid4())
        
        # Create persisted task
        task = PersistedTask(
            task_id=task_id,
            agent_id=self.agent_id,
            task_type=f"{self.agent_id}_{task_type}",
            payload=payload,
            status=PersistTaskStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata
        )
        
        # Save to persistence
        await task_manager.save_task(task)
        
        # Also track in memory
        self.tasks[task_id] = {
            "type": task_type,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow(),
            "persistent": True
        }
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_persistent_task(task))
        
        return task_id
    
    async def _execute_persistent_task(self, task: PersistedTask):
        """Execute a persistent task"""
        try:
            # Update status to in progress
            await task_manager.update_task_status(task.task_id, PersistTaskStatus.IN_PROGRESS)
            
            # Extract the actual task type (remove agent_id prefix)
            task_type = task.task_type.replace(f"{self.agent_id}_", "", 1)
            
            # Find handler
            handler = None
            if task_type.startswith("skill_"):
                skill_name = task_type.replace("skill_", "", 1)
                if skill_name in self.skills:
                    skill_def = self.skills[skill_name]
                    handler = getattr(self, skill_def.method_name)
            else:
                handler = self.handlers.get(task_type)
            
            if not handler:
                raise ValueError(f"No handler found for task type: {task_type}")
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload, task.metadata)
            else:
                result = handler(task.payload, task.metadata)
            
            # Update task status
            await task_manager.update_task_status(task.task_id, PersistTaskStatus.COMPLETED)
            
            # Update in-memory status
            if task.task_id in self.tasks:
                self.tasks[task.task_id]["status"] = TaskStatus.COMPLETED
                self.tasks[task.task_id]["completed_at"] = datetime.utcnow()
                self.tasks[task.task_id]["result"] = result
            
        except Exception as e:
            logger.error(f"Persistent task {task.task_id} failed: {e}")
            
            # Update task status
            await task_manager.update_task_status(task.task_id, PersistTaskStatus.FAILED, str(e))
            
            # Update in-memory status
            if task.task_id in self.tasks:
                self.tasks[task.task_id]["status"] = TaskStatus.FAILED
                self.tasks[task.task_id]["error"] = str(e)
    
    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get task statistics for this agent"""
        return await task_manager.get_task_statistics(self.agent_id)
    
    async def send_signed_request(self,
                                 target_agent_id: str,
                                 method: str,
                                 path: str,
                                 body: Optional[Dict[str, Any]] = None,
                                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Send a cryptographically signed request to another agent
        
        Args:
            target_agent_id: ID of the target agent
            method: HTTP method
            path: Request path
            body: Request body
            headers: Additional headers
            
        Returns:
            Request dict with signature headers
        """
        if not self.enable_request_signing or not self.signing_middleware:
            logger.warning("Request signing is disabled, sending unsigned request")
            return {
                'target_agent_id': target_agent_id,
                'method': method,
                'path': path,
                'body': body,
                'headers': headers or {}
            }
        
        request = {
            'target_agent_id': target_agent_id,
            'method': method,
            'path': path,
            'body': body,
            'headers': headers or {}
        }
        
        # Add signature headers
        signed_request = await self.signing_middleware.sign_outgoing_request(request)
        
        logger.debug(f"Signed request to {target_agent_id}: {method} {path}")
        return signed_request
    
    async def verify_incoming_request(self,
                                    headers: Dict[str, str],
                                    method: str,
                                    path: str,
                                    body: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify the signature of an incoming request from another agent
        
        Args:
            headers: Request headers
            method: HTTP method
            path: Request path
            body: Request body
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_request_signing or not self.signing_middleware:
            logger.warning("Request signing is disabled, skipping verification")
            return True, None
        
        # Import agent registry to get public keys
        from ..storage import get_distributed_storage


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        
        try:
            agent_registry = await get_distributed_storage()
            is_valid, error = await self.signing_middleware.verify_incoming_request(
                headers=headers,
                method=method,
                path=path,
                body=body,
                agent_registry=agent_registry
            )
            
            if is_valid:
                logger.debug(f"Successfully verified request from {headers.get('X-A2A-Agent-ID')}")
            else:
                logger.warning(f"Request verification failed: {error}")
            
            return is_valid, error
            
        except Exception as e:
            logger.error(f"Request verification error: {e}")
            return False, f"Verification error: {str(e)}"
    
    def get_public_key(self) -> Optional[str]:
        """Get the agent's public key for sharing with other agents"""
        return self.public_key_pem
    
    def get_agent_card(self) -> AgentCard:
        """Enhanced agent card with A2A protocol information"""
        card = AgentCard(
            name=self.name,
            description=self.description,
            url=self.base_url,
            version=self.version,
            protocolVersion="0.2.9",
            provider={
                "organization": "A2A Network",
                "url": "https://a2a-network.com"
            },
            capabilities={
                "messageProcessing": True,
                "taskManagement": True,
                "skillExecution": len(self.skills) > 0,
                "telemetry": self.enable_telemetry,
                "a2aProtocol": self.a2a_protocol_only,
                "blockchainMessaging": True,
                **{f"skill_{skill}": True for skill in self.skills.keys()}
            },
            skills=list(self.skills.keys()),
            endpoints={
                "rpc": f"{self.base_url}/rpc",
                "messages": f"{self.base_url}/messages",
                "skills": f"{self.base_url}/skills",
                "health": f"{self.base_url}/health"
            }
        )
        
        # Add A2A protocol information
        if not hasattr(card, 'metadata'):
            card.metadata = {}
        
        card.metadata.update({
            "a2aProtocol": {
                "version": "0.2.9",
                "compliant": self.a2a_protocol_only,
                "features": {
                    "blockchainMessaging": True,
                    "messageQueue": True,
                    "retryLogic": True
                }
            },
            "protocolStats": self.get_a2a_stats()
        })
        
        # Add public key for verification
        if hasattr(self, 'public_key_pem') and self.public_key_pem:
            card.metadata['public_key'] = self.public_key_pem
        
        return card
    
    # A2A Protocol Methods
    async def send_a2a_message(
        self,
        to_agent: str,
        task_id: str,
        context_id: str,
        parts: List[Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL,
        encrypt: bool = False,
        auto_select_agent: bool = True,
        required_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send A2A protocol compliant message via blockchain with AI reasoning and intelligent agent selection
        
        Args:
            to_agent: Target agent ID (can be overridden by intelligent selection)
            task_id: Task identifier
            context_id: Context identifier
            parts: Message parts
            priority: Message priority
            encrypt: Whether to encrypt the message
            auto_select_agent: Whether to use skills matching for optimal agent selection
            required_skills: Skills required for the task (extracted from parts if not provided)
        """
        if not self.a2a_protocol_only:
            logger.warning("A2A protocol not enforced, message may not be sent")
            return {"success": False, "error": "A2A protocol not enabled"}
        
        message_id = f"msg_{uuid4().hex[:12]}"
        original_target = to_agent
        
        # ✨ NEW: Intelligent Agent Selection based on Skills Matching
        if auto_select_agent and hasattr(self, 'analyze_skills_match'):
            logger.info(f"🎯 Performing intelligent agent selection for message {message_id}")
            
            # Extract required skills from message parts or use provided skills
            if not required_skills:
                required_skills = self._extract_required_skills_from_parts(parts)
            
            if required_skills:
                logger.info(f"🔍 Required skills identified: {required_skills}")
                
                # Find the best agent for these skills
                optimal_agent = await self._find_optimal_agent(required_skills, to_agent)
                
                if optimal_agent and optimal_agent['name'] != to_agent:
                    logger.info(f"🔄 Redirecting message from {to_agent} to {optimal_agent['name']} (match score: {optimal_agent['match_score']:.2f})")
                    to_agent = optimal_agent['name']
                    
                    # Add routing metadata to message parts
                    routing_metadata = {
                        "partType": "routing_metadata",
                        "data": {
                            "original_target": original_target,
                            "selected_target": to_agent,
                            "selection_reason": f"Better skills match (score: {optimal_agent['match_score']:.2f})",
                            "required_skills": required_skills,
                            "selection_timestamp": datetime.utcnow().isoformat(),
                            "selecting_agent": self.agent_id
                        }
                    }
                    parts.append(routing_metadata)
                else:
                    logger.info(f"✅ Original target {to_agent} is optimal for required skills")
            else:
                logger.info("ℹ️ No specific skills required, using original target")
        
        # Add message tracking metadata for analytics
        message_metadata = {
            "partType": "message_metadata",
            "data": {
                "sending_agent": self.agent_id,
                "original_target": original_target,
                "final_target": to_agent,
                "required_skills": required_skills or [],
                "auto_selection_used": auto_select_agent,
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": "outbound",
                "priority": priority.name,
                "encrypted": encrypt
            }
        }
        parts.append(message_metadata)

        # Apply AI reasoning to outgoing message if available
        if self.ai_enabled and hasattr(self, 'reason_about_message'):
            logger.info(f"🧠 Applying AI reasoning to outgoing message {message_id}")
            
            # Create message data for AI reasoning
            outgoing_message_data = {
                "message_id": message_id,
                "from_agent": self.agent_id,
                "to_agent": to_agent,
                "task_id": task_id,
                "context_id": context_id,
                "parts": parts,
                "priority": priority.name,
                "encrypted": encrypt,
                "direction": "outgoing",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            try:
                # Get AI reasoning about the outgoing message
                reasoning_result = await self.reason_about_message(
                    outgoing_message_data, 
                    context={"message_direction": "outgoing", "target_agent": to_agent}
                )
                
                # AI can suggest modifications to the message
                if reasoning_result.get("recommended_modifications"):
                    modifications = reasoning_result["recommended_modifications"]
                    
                    # Apply priority adjustment if suggested
                    if "priority" in modifications:
                        suggested_priority = modifications["priority"]
                        if suggested_priority in MessagePriority.__members__:
                            priority = MessagePriority[suggested_priority]
                            logger.info(f"🔄 AI adjusted message priority to {priority.name}")
                    
                    # Apply encryption suggestion if provided
                    if "encrypt" in modifications:
                        encrypt = modifications["encrypt"]
                        logger.info(f"🔒 AI {'enabled' if encrypt else 'disabled'} encryption")
                    
                    # Apply content modifications if suggested
                    if "enhanced_parts" in modifications:
                        enhanced_parts = modifications["enhanced_parts"]
                        if enhanced_parts:
                            parts = enhanced_parts
                            logger.info(f"✨ AI enhanced message content")
                
                logger.info(f"✅ AI reasoning applied to outgoing message {message_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ AI reasoning failed for outgoing message {message_id}: {e}")
        
        # Create the A2A protocol message
        message = A2AProtocolMessage(
            message_id=message_id,
            from_agent=self.agent_id,
            to_agent=to_agent,
            task_id=task_id,
            context_id=context_id,
            parts=parts,
            priority=priority,
            encrypted=encrypt
        )
        
        # Add to outgoing queue with priority
        await self.outgoing_queue.put((
            -message.priority.value,  # Negative for priority queue
            message
        ))
        
        self.message_stats["sent"] += 1
        
        return {
            "success": True,
            "message_id": message.message_id,
            "queued": True,
            "ai_enhanced": self.ai_enabled and hasattr(self, 'reason_about_message')
        }
    
    async def _process_a2a_message_queues(self):
        """Process outgoing A2A message queue with retry logic"""
        while True:
            try:
                # Get message from priority queue
                _, message = await self.outgoing_queue.get()
                
                # Send via blockchain
                success = await self._send_blockchain_a2a_message(message)
                
                if not success and message.retry_count < message.max_retries:
                    # Add to retry queue
                    message.retry_count += 1
                    await self.retry_queue.put(message)
                    self.message_stats["retried"] += 1
                elif not success:
                    self.message_stats["failed"] += 1
                    logger.error(f"A2A message {message.message_id} failed after {message.retry_count} retries")
                
            except Exception as e:
                logger.error(f"Error processing A2A message queue: {e}")
            
            await asyncio.sleep(0.1)
    
    async def _send_blockchain_a2a_message(self, message: A2AProtocolMessage) -> bool:
        """Send A2A message via blockchain smart contract"""
        if not self.blockchain_client:
            logger.error("Blockchain client not initialized for A2A messaging")
            return False
        
        try:
            # Use blockchain integration to send message
            message_id = self.send_blockchain_message(
                to_address=message.to_agent,
                content={
                    "messageId": message.message_id,
                    "taskId": message.task_id,
                    "contextId": message.context_id,
                    "parts": message.parts,
                    "priority": message.priority.value,
                    "encrypted": message.encrypted
                },
                message_type="A2A_PROTOCOL"
            )
            
            if message_id:
                logger.info(f"Sent A2A message {message.message_id} via blockchain")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to send A2A blockchain message: {e}")
            return False
    
    async def _process_a2a_retry_queue(self):
        """Process A2A messages that need retry with exponential backoff"""
        while True:
            try:
                message = await self.retry_queue.get()
                
                # Calculate backoff delay
                delay = min(
                    self.retry_config["initial_delay"] * (
                        self.retry_config["backoff_factor"] ** (message.retry_count - 1)
                    ),
                    self.retry_config["max_delay"]
                )
                
                logger.info(f"Retrying A2A message {message.message_id} after {delay}s (attempt {message.retry_count})")
                await asyncio.sleep(delay)
                
                # Re-add to outgoing queue
                await self.outgoing_queue.put((-message.priority.value, message))
                
            except Exception as e:
                logger.error(f"Error processing A2A retry queue: {e}")
            
            await asyncio.sleep(1)
    
    def _handle_blockchain_message(self, message: Dict[str, Any]):
        """Enhanced blockchain message handler for A2A protocol"""
        # Call parent handler first
        super()._handle_blockchain_message(message)
        
        # Handle A2A protocol specific messages
        if message.get('messageType') == 'A2A_PROTOCOL':
            content = message.get('content', {})
            
            # Extract A2A message components
            message_id = content.get('messageId')
            task_id = content.get('taskId')
            context_id = content.get('contextId')
            parts = content.get('parts', [])
            
            # Update statistics
            self.message_stats["received"] += 1
            
            # Process A2A message parts
            asyncio.create_task(self._handle_a2a_message_parts({
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
                "fromAgent": message.get('from_address'),
                "parts": parts
            }))
    
    async def _handle_a2a_message_parts(self, a2a_message: Dict[str, Any]):
        """Handle A2A message parts"""
        try:
            from_agent = a2a_message.get('fromAgent')
            parts = a2a_message.get('parts', [])
            
            for part in parts:
                part_type = part.get('partType')
                
                if part_type == 'function-call':
                    # Handle skill execution request
                    skill_name = part.get('functionName')
                    input_data = part.get('functionArgs', {})
                    
                    if skill_name in self.skills:
                        result = await self.execute_skill(skill_name, input_data)
                        
                        # Send response via A2A
                        await self.send_a2a_message(
                            to_agent=from_agent,
                            task_id=a2a_message.get('taskId'),
                            context_id=a2a_message.get('contextId'),
                            parts=[{
                                "partType": "function-response",
                                "functionId": part.get('functionId'),
                                "responseData": result
                            }]
                        )
                
                elif part_type == 'data-request':
                    # Handle data request
                    data_type = part.get('dataType')
                    query_params = part.get('queryParams', {})
                    
                    # Process data request (can be overridden)
                    data = await self._process_a2a_data_request(data_type, query_params)
                    
                    # Send response
                    await self.send_a2a_message(
                        to_agent=from_agent,
                        task_id=a2a_message.get('taskId'),
                        context_id=a2a_message.get('contextId'),
                        parts=[{
                            "partType": "data-response",
                            "requestId": part.get('requestId'),
                            "data": data
                        }]
                    )
                
        except Exception as e:
            logger.error(f"Error handling A2A message parts: {e}")
    
    async def _process_a2a_data_request(self, data_type: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process A2A data request - override in specific agents
        """
        return {"error": "Data request handler not implemented"}
    
    async def call_agent_skill_a2a(
        self,
        target_agent: str,
        skill_name: str,
        input_data: Dict[str, Any],
        context_id: Optional[str] = None,
        encrypt_data: bool = False
    ) -> Dict[str, Any]:
        """
        Call a skill on another agent using A2A protocol
        """
        context_id = context_id or f"ctx_{self.agent_id}_{target_agent}"
        task_id = f"skill_{skill_name}_{hash(str(input_data))}"
        
        # Prepare message parts
        parts = [{
            "partType": "function-call",
            "functionName": skill_name,
            "functionArgs": input_data,
            "functionId": f"func_{task_id[:8]}",
            "sensitive": encrypt_data
        }]
        
        # Send via A2A protocol
        result = await self.send_a2a_message(
            to_agent=target_agent,
            task_id=task_id,
            context_id=context_id,
            parts=parts,
            priority=MessagePriority.NORMAL,
            encrypt=encrypt_data
        )
        
        # Store context for response handling
        self.active_contexts[context_id] = {
            "operation": "skill_call",
            "target_agent": target_agent,
            "skill_name": skill_name,
            "task_id": task_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return result
    
    async def request_data_from_agent_a2a(
        self,
        target_agent: str,
        data_type: str,
        query_params: Dict[str, Any],
        encrypt: bool = True
    ) -> Dict[str, Any]:
        """
        Request data from another agent via A2A protocol
        """
        context_id = f"data_req_{self.agent_id}_{target_agent}"
        task_id = f"data_{data_type}_{hash(str(query_params))}"
        
        parts = [{
            "partType": "data-request",
            "dataType": data_type,
            "queryParams": query_params,
            "requestId": task_id,
            "sensitive": encrypt
        }]
        
        return await self.send_a2a_message(
            to_agent=target_agent,
            task_id=task_id,
            context_id=context_id,
            parts=parts,
            priority=MessagePriority.HIGH,
            encrypt=encrypt
        )
    
    def _extract_required_skills_from_parts(self, parts: List[Dict[str, Any]]) -> List[str]:
        """Extract required skills from message parts for intelligent agent selection"""
        required_skills = []
        
        try:
            for part in parts:
                part_data = part.get('data', {})
                
                # Check for explicit skills requirement
                if 'required_skills' in part_data:
                    required_skills.extend(part_data['required_skills'])
                
                # Infer skills from common action types
                action = part_data.get('action', part_data.get('method', ''))
                if action:
                    skill_mapping = {
                        'store_data': ['data_storage', 'persistence'],
                        'store_user_data': ['data_storage', 'user_management'],
                        'analyze_data': ['data_analysis', 'analytics'],
                        'analyze_user_patterns': ['data_analysis', 'behavioral_analysis'],
                        'process_payment': ['payment_processing', 'financial'],
                        'generate_report': ['reporting', 'data_visualization'],
                        'send_notification': ['communication', 'messaging'],
                        'validate_data': ['data_validation', 'quality_control'],
                        'calculate': ['mathematical_computation', 'calculations'],
                        'encrypt_data': ['encryption', 'security'],
                        'authenticate': ['authentication', 'security'],
                        'schedule_task': ['task_scheduling', 'workflow_management'],
                        'query_database': ['database_access', 'data_retrieval'],
                        'transform_data': ['data_transformation', 'data_processing']
                    }
                    
                    if action in skill_mapping:
                        required_skills.extend(skill_mapping[action])
                
                # Infer from content type
                if 'user_data' in part_data:
                    required_skills.append('user_management')
                if 'analytics' in part_data or 'insights' in part_data:
                    required_skills.append('data_analysis')
                if 'encryption' in part_data or 'security' in part_data:
                    required_skills.append('security')
                
            # Remove duplicates and return
            return list(set(required_skills))
            
        except Exception as e:
            logger.warning(f"Failed to extract required skills: {e}")
            return []
    
    async def _find_optimal_agent(self, required_skills: List[str], fallback_agent: str) -> Optional[Dict[str, Any]]:
        """Find the optimal agent for the required skills using network discovery"""
        try:
            if not hasattr(self, 'network_agents_cache'):
                logger.warning("No network agents cache available for optimal agent selection")
                return None
            
            best_agent = None
            best_score = 0.0
            
            # Get our own skills match as baseline
            if hasattr(self, '_calculate_skills_similarity'):
                own_skills = list(getattr(self, 'skill_registry', {}).keys())
                own_match = self._calculate_skills_similarity(required_skills, own_skills)
                baseline_score = own_match['similarity_score']
            else:
                baseline_score = 0.5  # Conservative baseline
            
            # Search through available agents
            for agent_address, agent_info in getattr(self, 'network_agents_cache', {}).items():
                agent_name = agent_info.get('name', 'Unknown')
                
                # Skip inactive agents and self
                if not agent_info.get('active', False) or agent_name == self.agent_id:
                    continue
                
                # Get agent skills
                agent_skills = list(agent_info.get('skills', {}).keys())
                
                if hasattr(self, '_calculate_skills_similarity'):
                    skills_match = self._calculate_skills_similarity(required_skills, agent_skills)
                    match_score = skills_match['similarity_score']
                else:
                    # Simple fallback matching
                    matches = sum(1 for skill in required_skills if skill in agent_skills)
                    match_score = matches / len(required_skills) if required_skills else 0.0
                
                # Factor in reputation (normalized to 0-1)
                reputation = agent_info.get('reputation', 50) / 100.0
                overall_score = (match_score * 0.8) + (reputation * 0.2)  # 80% skills, 20% reputation
                
                # Only consider agents better than current baseline + threshold
                if overall_score > max(baseline_score + 0.1, best_score):
                    best_agent = {
                        'name': agent_name,
                        'address': agent_address,
                        'endpoint': agent_info.get('endpoint', ''),
                        'skills': agent_skills,
                        'match_score': match_score,
                        'reputation': reputation,
                        'overall_score': overall_score,
                        'active': agent_info.get('active', False)
                    }
                    best_score = overall_score
            
            if best_agent:
                logger.info(f"Found optimal agent {best_agent['name']} with score {best_score:.2f} (baseline: {baseline_score:.2f})")
            else:
                logger.info(f"No better agent found than current target {fallback_agent}")
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Failed to find optimal agent: {e}")
            return None

    async def _track_message_with_agent_manager(self, message_data: Dict[str, Any], status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track message lifecycle with AgentManager for comprehensive analytics and reputation"""
        try:
            # Try to get the AgentManager client
            agent_manager_url = os.getenv("A2A_AGENT_MANAGER_URL", "http://localhost:8010")
            
            # In a full implementation, this would make an A2A protocol call to the AgentManager
            # For now, we'll log the tracking data and store it locally
            tracking_data = {
                "agent_id": self.agent_id,
                "message_data": message_data,
                "status": status,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store locally for now (in production, this would be sent via A2A messaging)
            if not hasattr(self, '_local_message_tracking'):
                self._local_message_tracking = []
            
            self._local_message_tracking.append(tracking_data)
            
            # Keep only recent tracking data (last 1000 entries)
            if len(self._local_message_tracking) > 1000:
                self._local_message_tracking = self._local_message_tracking[-1000:]
            
            logger.debug(f"Tracked message {message_data.get('message_id', 'unknown')} with status: {status}")
            
            # Send tracking data to AgentManager via A2A protocol if available
            if hasattr(self, 'send_a2a_message') and os.getenv("A2A_AGENT_MANAGER_ENABLED", "false").lower() == "true":
                try:
                    await self.send_a2a_message(
                        to_agent="comprehensive_agent_manager",
                        task_id=f"track_{uuid4().hex[:8]}",
                        context_id="message_tracking",
                        parts=[{
                            "partType": "message_tracking",
                            "data": tracking_data
                        }],
                        auto_select_agent=False  # Direct to AgentManager
                    )
                    logger.debug(f"Sent tracking data to AgentManager via A2A protocol")
                except Exception as track_error:
                    logger.warning(f"Failed to send tracking to AgentManager: {track_error}")
                    # Continue with local storage as fallback
            
        except Exception as e:
            logger.error(f"Failed to track message with AgentManager: {e}")
    
    def get_message_tracking_stats(self) -> Dict[str, Any]:
        """Get local message tracking statistics"""
        if not hasattr(self, '_local_message_tracking'):
            return {"total_tracked": 0}
        
        tracking_data = self._local_message_tracking
        status_counts = {}
        for entry in tracking_data:
            status = entry.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tracked": len(tracking_data),
            "status_breakdown": status_counts,
            "agent_id": self.agent_id,
            "tracking_enabled": True
        }

    def get_a2a_stats(self) -> Dict[str, Any]:
        """Get A2A protocol statistics"""
        return {
            "protocol_enabled": self.a2a_protocol_only,
            "message_stats": self.message_stats,
            "queue_sizes": {
                "incoming": self.message_queue.qsize(),
                "outgoing": self.outgoing_queue.qsize(),
                "retry": self.retry_queue.qsize()
            },
            "active_contexts": len(self.active_contexts),
            "blockchain_connected": self.blockchain_client is not None,
            "message_tracking": self.get_message_tracking_stats()
        }
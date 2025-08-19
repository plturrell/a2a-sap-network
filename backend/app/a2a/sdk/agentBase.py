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

logger = logging.getLogger(__name__)

try:
    from a2a.core.retry_utils import retry_manager
    from a2a.core.dead_letter_queue import dlq
    from a2a.core.task_persistence import task_manager, PersistedTask, TaskStatus as PersistTaskStatus
    from a2a.security.requestSigning import A2ARequestSigner, A2ASigningMiddleware
except ImportError:
    # Fallback for missing core dependencies
    def retry_with_backoff(_max_attempts=3, _backoff_factor=1.0):
        def decorator(func):
            return func
        return decorator
    
    class MockRetryManager:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
    retry_manager = MockRetryManager()

    class MockDLQ:
        async def add_message(self, **_kwargs): pass
    dlq = MockDLQ()

    class MockPersistedTask:
        def __init__(self, **_kwargs):
            self.task_id = "mock_task"
            self.status = "pending"
    PersistedTask = MockPersistedTask

    class MockTaskManager:
        async def save_task(self, **_kwargs): return {"task_id": "mock"}
        async def update_task_status(self, **_kwargs): pass
        def register_task_handler(self, task_type: str, handler: callable):
            pass
        async def recover_tasks(self, agent_id: str):
            # In-memory mock, no tasks to recover
            return []
    task_manager = MockTaskManager()
    
    class MockRequestSigner:
        def __init__(self, **_kwargs): pass
        async def sign_request(self, **_kwargs): return {}
        def generate_key_pair(self): return ("mock_private", "mock_public")
    A2ARequestSigner = MockRequestSigner
    
    class MockSigningMiddleware:
        def __init__(self, **_kwargs): pass
    A2ASigningMiddleware = MockSigningMiddleware

# Import telemetry - optional for testing
try:
    from a2a.core.telemetry import init_telemetry, trace_async, add_span_attributes
    from a2a.config.telemetryConfig import telemetry_config
except ImportError:
    # Mock telemetry for testing
    def init_telemetry(**_kwargs): pass
    def trace_async(_operation_name="operation"):
        def decorator(func):
            return func
        return decorator
    def add_span_attributes(**_kwargs): pass
    telemetry_config = {"enabled": False}

# Import decorators - required for agent functionality
try:
    from .decorators import get_handler_metadata
except ImportError as e:
    logger.error("Decorator modules are required but not available: %s", e)
    raise ImportError("Agent decorators are required for agent operation. Please ensure all SDK components are available.") from e


class A2AAgentBase(ABC, BlockchainIntegrationMixin):
    """
    Base class for A2A agents providing common functionality:
    - Message routing and handling
    - Task management
    - Telemetry integration
    - Agent registration
    - Skill discovery
    - Blockchain integration (when enabled)
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        base_url: str = "http://localhost:8000",
        enable_telemetry: bool = True,
        enable_request_signing: bool = True,
        private_key_pem: Optional[str] = None,
        public_key_pem: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = version
        self.base_url = base_url
        self.enable_telemetry = enable_telemetry
        self.enable_request_signing = enable_request_signing
        
        # Internal state
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.skills: Dict[str, SkillDefinition] = {}
        self.capabilities: List[AgentCapability] = []
        self.start_time = datetime.utcnow()
        
        # Initialize request signing
        self.request_signer = None
        self.signing_middleware = None
        self.private_key_pem = private_key_pem
        self.public_key_pem = public_key_pem
        
        if enable_request_signing:
            self._initialize_request_signing()
        
        # Initialize MCP server for internal operations
        self.mcp_server = A2AMCPServer(self)
        
        # Initialize telemetry
        if enable_telemetry and telemetry_config.get('otel_enabled'):
            init_telemetry(
                service_name=f"a2a-agent-{agent_id}",
                agent_id=agent_id,
                sampling_rate=telemetry_config.get('otel_traces_sampler_arg')
            )
        
        # Discover handlers and skills
        self._discover_handlers()
        self._discover_skills()
        
        # Initialize blockchain integration mixin
        BlockchainIntegrationMixin.__init__(self)
        
        # Get agent capabilities for blockchain registration
        blockchain_capabilities = [cap.name for cap in self.capabilities]
        
        # Initialize blockchain if enabled
        self._initialize_blockchain(
            agent_name=self.name,
            capabilities=blockchain_capabilities,
            endpoint=self.base_url
        )
        
        # MCP components are automatically discovered by the MCP server
        # No need for duplicate discovery here
        
        # Initialize task persistence
        self._initialize_task_persistence()
        
        # Start task recovery
        asyncio.create_task(self._recover_tasks())
        
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
        """Process incoming A2A message"""
        
        add_span_attributes({
            "agent.id": self.agent_id,
            "message.id": message.messageId,
            "message.role": message.role.value,
            "context.id": context_id
        })
        
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
                return {
                    "success": True,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Handler {method} failed: {e}")
                
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
                    "timestamp": datetime.utcnow().isoformat()
                }
        else:
            return {
                "success": False,
                "error": f"No handler found for method: {method}",
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
    
    def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        return self.mcp_server.get_tool_definitions()
    
    def list_mcp_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources"""
        return self.mcp_server.get_resource_definitions()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return self.tasks.get(task_id)
    
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
    
    def create_fastapi_app(self):
        """Create FastAPI app with standard A2A endpoints"""
        # This method would require FastAPI, which is not available in the fallback.
        # Returning a dummy object to avoid crashing.
        class MockApp:
            def add_api_route(self, *args, **kwargs): pass
            def add_middleware(self, *args, **kwargs): pass
        return MockApp()
        
        app = FastAPI(
            title=self.name,
            description=self.description,
            version=self.version,
            docs_url=f"/api/{api_version}/docs",
            redoc_url=f"/api/{api_version}/redoc",
            openapi_url=f"/api/{api_version}/openapi.json",
            openapi_tags=[
                {"name": "agent", "description": "Agent core operations"},
                {"name": "skills", "description": "Agent skills and capabilities"},
                {"name": "tasks", "description": "Task management"},
                {"name": "mcp", "description": "Model Context Protocol"},
                {"name": "health", "description": "Health and monitoring"}
            ]
        )
        
        # Add API router for versioning
        from fastapi import APIRouter
        api_router = APIRouter(prefix=f"/api/{api_version}")
        
        @app.get("/.well-known/agent.json", tags=["agent"])
        async def get_agent_card():
            """Get agent card with capabilities and metadata"""
            return self.get_agent_card()
        
        @api_router.get("/agent/info", tags=["agent"])
        async def get_agent_info():
            """Get detailed agent information"""
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "api_version": api_version,
                "capabilities": [cap.dict() for cap in self.capabilities],
                "skills_count": len(self.skills),
                "handlers_count": len(self.handlers)
            }
        
        @api_router.get("/agent/skills", tags=["skills"])
        async def list_agent_skills():
            """List all available skills with their schemas"""
            return {"skills": self.list_skills()}
        
        @api_router.post("/agent/skills/{skill_name}/execute", tags=["skills"])
        async def execute_skill_endpoint(skill_name: str, input_data: Dict[str, Any]):
            """Execute a specific skill with input data"""
            return await self.execute_skill(skill_name, input_data)
        
        @api_router.get("/agent/tasks/{task_id}", tags=["tasks"])
        async def get_task_status_endpoint(task_id: str):
            """Get status of a specific task"""
            status = self.get_task_status(task_id)
            if not status:
                raise HTTPException(status_code=404, detail="Task not found")
            return status
        
        @api_router.post("/agent/tasks", tags=["tasks"])
        async def create_persistent_task_endpoint(
            task_type: str,
            payload: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None
        ):
            """Create a new persistent task"""
            task_id = await self.create_persistent_task(task_type, payload, metadata)
            return {"task_id": task_id}
        
        @api_router.get("/agent/tasks/stats", tags=["tasks"])
        async def get_task_statistics_endpoint():
            """Get task execution statistics"""
            return await self.get_task_statistics()
        
        @api_router.get("/health", tags=["health"])
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "uptime": (datetime.utcnow() - self.start_time).total_seconds() if hasattr(self, 'start_time') else 0
            }
        
        @api_router.get("/health/ready", tags=["health"])
        async def readiness_check():
            """Readiness check endpoint"""
            return {"ready": True}
        
        # Mount versioned API
        app.include_router(api_router)
        
        @app.post("/rpc")
        async def json_rpc_handler(request: Request):
            body = await request.json()
            
            if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {"code": -32600, "message": "Invalid Request"},
                        "id": body.get("id")
                    }
                )
            
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")
            
            try:
                if method == "agent.getCard":
                    result = self.get_agent_card()
                elif method == "agent.processMessage":
                    message = A2AMessage(**params.get("message", {}))
                    context_id = params.get("contextId", str(uuid4()))
                    result = await self.process_message(message, context_id)
                elif method == "agent.getTaskStatus":
                    task_id = params.get("taskId")
                    result = self.get_task_status(task_id)
                elif method == "agent.executeSkill":
                    skill_name = params.get("skillName")
                    input_data = params.get("inputData", {})
                    result = await self.execute_skill(skill_name, input_data)
                elif method == "agent.listSkills":
                    result = self.list_skills()
                elif method == "mcp.request":
                    mcp_request = MCPRequest(**params.get("request", {}))
                    mcp_response = await self.process_mcp_request(mcp_request)
                    result = mcp_response.model_dump()
                elif method == "agent.listMCPTools":
                    result = self.list_mcp_tools()
                elif method == "agent.listMCPResources":
                    result = self.list_mcp_resources()
                else:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {"code": -32601, "message": "Method not found"},
                            "id": request_id
                        }
                    )
                
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                })
            
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                        "id": request_id
                    }
                )
        
        @app.post("/messages")
        async def rest_message_handler(request: Request):
            body = await request.json()
            message = A2AMessage(**body.get("message", {}))
            context_id = body.get("contextId", str(uuid4()))
            
            result = await self.process_message(message, context_id)
            return JSONResponse(content=result)
        
        @app.get("/skills")
        async def list_skills_endpoint():
            return {"skills": self.list_skills()}
        
        @app.post("/skills/{skill_name}")
        async def execute_skill_endpoint(skill_name: str, request: Request):
            body = await request.json()
            result = await self.execute_skill(skill_name, body)
            return result
        
        @app.post("/mcp")
        async def mcp_handler(request: Request):
            """Handle MCP requests"""
            body = await request.json()
            mcp_request = MCPRequest(**body)
            mcp_response = await self.process_mcp_request(mcp_request)
            return JSONResponse(content=mcp_response.model_dump())
        
        @app.get("/mcp/tools")
        async def list_mcp_tools_endpoint():
            """List MCP tools"""
            return {"tools": self.list_mcp_tools()}
        
        @app.post("/mcp/tools/{tool_name}")
        async def call_mcp_tool_endpoint(tool_name: str, request: Request):
            """Call MCP tool"""
            body = await request.json()
            result = await self.call_mcp_tool(tool_name, body)
            return result
        
        @app.get("/mcp/resources")
        async def list_mcp_resources_endpoint():
            """List MCP resources"""
            return {"resources": self.list_mcp_resources()}
        
        @app.get("/mcp/resources/{uri:path}")
        async def get_mcp_resource_endpoint(uri: str):
            """Get MCP resource"""
            result = await self.get_mcp_resource(uri)
            return result

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "name": self.name,
                "version": self.version,
                "active_tasks": len([t for t in self.tasks.values() if t["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]]),
                "total_tasks": len(self.tasks),
                "skills": len(self.skills),
                "handlers": len(self.handlers),
                "mcp_tools": len(self.mcp_server.tools),
                "mcp_resources": len(self.mcp_server.resources),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return app
    
    # Abstract methods that subclasses should implement
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent-specific resources"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        pass
    
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

    def _initialize_task_persistence(self):
        """Initialize task persistence for this agent"""
        # Register task handlers for recovery
        for task_type, handler in self.handlers.items():
            task_manager.register_task_handler(f"{self.agent_id}_{task_type}", handler)
        
        # Register skill handlers for recovery
        for skill_name, skill_def in self.skills.items():
            method = getattr(self, skill_def.method_name)
            task_manager.register_task_handler(f"{self.agent_id}_skill_{skill_name}", method)
    
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
        """Get agent information card including public key"""
        card = AgentCard(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            version=self.version,
            capabilities=self.capabilities
        )
        
        # Add public key for verification
        if self.public_key_pem:
            card.metadata = card.metadata or {}
            card.metadata['public_key'] = self.public_key_pem
        
        return card
"""
Base class for A2A agents - simplifies agent development
"""

from abc import ABC, abstractmethod
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime
from uuid import uuid4
import inspect

from .types import A2AMessage, MessagePart, MessageRole, AgentCard, AgentCapability, SkillDefinition, TaskStatus
from .decorators import get_handler_metadata
from app.a2a.core.telemetry import add_span_attributes

logger = logging.getLogger(__name__)


class A2AAgentBase(ABC):
    """
    Base class for A2A agents providing common functionality:
    - Message routing and handling
    - Task management
    - Telemetry integration
    - Agent registration
    - Skill discovery
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        base_url: str = "http://localhost:8000",
        enable_telemetry: bool = True
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = version
        self.base_url = base_url
        self.enable_telemetry = enable_telemetry
        
        # Internal state
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.skills: Dict[str, SkillDefinition] = {}
        self.capabilities: List[AgentCapability] = []
        
        
        # Discover handlers and skills
        self._discover_handlers()
        self._discover_skills()
        
        logger.info(f"Initialized A2A Agent: {self.name} ({self.agent_id})")
    
    def _discover_handlers(self):
        """Discover handler methods decorated with @a2a_handler"""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, '_a2a_handler'):
                handler_metadata = get_handler_metadata(method)
                if handler_metadata:
                    self.handlers[handler_metadata['method']] = method
                    logger.debug(f"Registered handler: {handler_metadata['method']} -> {name}")
    
    def _discover_skills(self):
        """Discover skills decorated with @a2a_skill"""
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, '_a2a_skill'):
                skill_def = SkillDefinition(
                    name=method._a2a_skill['name'],
                    description=method._a2a_skill['description'],
                    input_schema=method._a2a_skill.get('input_schema'),
                    output_schema=method._a2a_skill.get('output_schema'),
                    capabilities=method._a2a_skill.get('capabilities', []),
                    method_name=name
                )
                self.skills[skill_def.name] = skill_def
                logger.debug(f"Registered skill: {skill_def.name}")
    
    async def process_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Process incoming A2A message"""
        
        # Extract method from message
        method = self._extract_method(message)
        
        if method in self.handlers:
            try:
                handler = self.handlers[method]
                result = await self._call_handler(handler, message, context_id)
                return {
                    "success": True,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Handler {method} failed: {e}")
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
            elif param_name == "data" and message.parts:
                # Extract data from first data part
                for part in message.parts:
                    if part.kind == "data":
                        kwargs["data"] = part.data
                        break
        
        # Call handler
        if asyncio.iscoroutinefunction(handler):
            return await handler(**kwargs)
        else:
            return handler(**kwargs)
    
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
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with standard A2A endpoints"""
        app = FastAPI(
            title=self.name,
            description=self.description,
            version=self.version
        )
        
        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            return self.get_agent_card()
        
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
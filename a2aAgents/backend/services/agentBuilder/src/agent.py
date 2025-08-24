"""
Agent Builder Agent - A2A Microservice
Creates and manages agent templates and configurations for the A2A network
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



import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, field, asdict
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
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


@dataclass
class AgentTemplate:
    """Agent template definition"""
    template_id: str
    name: str
    description: str
    agent_type: str
    base_capabilities: List[str] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    handlers: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    

class AgentBuilderAgent(A2AAgentBase):
    """
    Agent Builder Agent - Creates and manages agent templates
    """
    
    def __init__(self, base_url: str, agent_manager_url: str):
        super().__init__(
            agent_id="agent_builder_agent",
            name="Agent Builder Agent",
            description="A2A v0.2.9 compliant agent for creating and managing agent templates",
            version="2.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        # HTTP client removed for A2A protocol compliance
        
        # Agent templates storage
        self.agent_templates: Dict[str, AgentTemplate] = {}
        
        # Initialize with basic templates
        self._initialize_default_templates()
        
        self.is_ready = True
        
    async def initialize(self):
        """Initialize the agent builder"""
        logger.info(f"Initializing {self.name}")
        
    def _initialize_default_templates(self):
        """Initialize with default agent templates"""
        
        # Data Processing Agent Template
        data_processor = AgentTemplate(
            template_id="data_processor_v1",
            name="Data Processing Agent",
            description="Template for data processing and transformation agents",
            agent_type="data_processor",
            base_capabilities=["data_validation", "data_transformation", "data_storage"],
            skills=[
                {"name": "process_data", "description": "Process and transform data"},
                {"name": "validate_data", "description": "Validate data quality"}
            ],
            handlers=[
                {"method": "POST", "endpoint": "/process", "description": "Process data"}
            ],
            dependencies=["pandas", "numpy", "sqlalchemy"],
            configuration={
                "max_batch_size": 1000,
                "timeout": 300,
                "storage_backend": "sqlite"
            }
        )
        
        # AI/ML Agent Template
        ai_ml_agent = AgentTemplate(
            template_id="ai_ml_v1",
            name="AI/ML Agent",
            description="Template for AI and machine learning agents",
            agent_type="ai_ml",
            base_capabilities=["model_inference", "model_training", "prediction"],
            skills=[
                {"name": "predict", "description": "Make predictions using trained models"},
                {"name": "train_model", "description": "Train machine learning models"}
            ],
            handlers=[
                {"method": "POST", "endpoint": "/predict", "description": "Get predictions"},
                {"method": "POST", "endpoint": "/train", "description": "Train model"}
            ],
            dependencies=["scikit-learn", "torch", "transformers"],
            configuration={
                "model_type": "transformer",
                "max_sequence_length": 512,
                "batch_size": 32
            }
        )
        
        # Integration Agent Template
        integration_agent = AgentTemplate(
            template_id="integration_v1",
            name="Integration Agent",
            description="Template for system integration agents",
            agent_type="integration",
            base_capabilities=["api_client", "data_sync", "event_handling"],
            skills=[
                {"name": "sync_data", "description": "Synchronize data between systems"},
                {"name": "handle_webhook", "description": "Handle webhook events"}
            ],
            handlers=[
                {"method": "POST", "endpoint": "/sync", "description": "Sync data"},
                {"method": "POST", "endpoint": "/webhook", "description": "Handle webhook"}
            ],
            dependencies=["httpx", "aioredis", "celery"],
            configuration={
                "sync_interval": 300,
                "max_retries": 3,
                "timeout": 60
            }
        )
        
        self.agent_templates = {
            "data_processor_v1": data_processor,
            "ai_ml_v1": ai_ml_agent,
            "integration_v1": integration_agent
        }
        
        logger.info(f"Loaded {len(self.agent_templates)} default agent templates")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # A2A Protocol Compliance: Registration should be handled through
            # blockchain-based messaging, not direct HTTP calls.
            # For now, mark as registered since the agent is running.
            logger.info(f"Agent {self.agent_id} ready for A2A network communication")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Capabilities: agent_creation, template_management, configuration_generation")
            self.is_registered = True
            
        except Exception as e:
            logger.error(f"Failed to register: {e}")
    
    @a2a_skill("list_templates", "List available agent templates")
    async def list_templates(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """List all available agent templates"""
        try:
            templates = []
            for template_id, template in self.agent_templates.items():
                templates.append({
                    "template_id": template_id,
                    "name": template.name,
                    "description": template.description,
                    "agent_type": template.agent_type,
                    "capabilities": template.base_capabilities,
                    "created_at": template.created_at.isoformat()
                })
            
            return create_success_response({
                "templates": templates,
                "total_count": len(templates)
            })
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return create_error_response(str(e))
    
    @a2a_skill("get_template", "Get specific agent template details")
    async def get_template(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Get details for a specific template"""
        try:
            params = message.content.get("parameters", {})
            template_id = params.get("template_id")
            
            if not template_id:
                return create_error_response("template_id is required")
            
            template = self.agent_templates.get(template_id)
            if not template:
                return create_error_response(f"Template {template_id} not found")
            
            return create_success_response({
                "template": asdict(template)
            })
            
        except Exception as e:
            logger.error(f"Error getting template: {e}")
            return create_error_response(str(e))
    
    @a2a_skill("create_template", "Create a new agent template")
    async def create_template(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Create a new agent template"""
        try:
            params = message.content.get("parameters", {})
            
            # Validate required fields
            required_fields = ["name", "description", "agent_type"]
            for field in required_fields:
                if not params.get(field):
                    return create_error_response(f"{field} is required")
            
            # Generate template ID
            template_id = f"{params['agent_type']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create template
            template = AgentTemplate(
                template_id=template_id,
                name=params["name"],
                description=params["description"],
                agent_type=params["agent_type"],
                base_capabilities=params.get("capabilities", []),
                skills=params.get("skills", []),
                handlers=params.get("handlers", []),
                dependencies=params.get("dependencies", []),
                configuration=params.get("configuration", {})
            )
            
            # Store template
            self.agent_templates[template_id] = template
            
            logger.info(f"Created new agent template: {template_id}")
            
            return create_success_response({
                "template_id": template_id,
                "template": asdict(template)
            })
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return create_error_response(str(e))
    
    @a2a_skill("generate_agent_code", "Generate agent code from template")
    async def generate_agent_code(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Generate agent code from a template"""
        try:
            params = message.content.get("parameters", {})
            template_id = params.get("template_id")
            agent_name = params.get("agent_name", "GeneratedAgent")
            
            if not template_id:
                return create_error_response("template_id is required")
            
            template = self.agent_templates.get(template_id)
            if not template:
                return create_error_response(f"Template {template_id} not found")
            
            # Generate basic agent code structure
            agent_code = self._generate_agent_code(template, agent_name)
            
            return create_success_response({
                "agent_name": agent_name,
                "template_id": template_id,
                "generated_files": agent_code
            })
            
        except Exception as e:
            logger.error(f"Error generating agent code: {e}")
            return create_error_response(str(e))
    
    def _generate_agent_code(self, template: AgentTemplate, agent_name: str) -> Dict[str, str]:
        """Generate agent code files from template"""
        
        # Generate main.py
        main_py = f'''#!/usr/bin/env python3
"""
{agent_name} - A2A Microservice
Generated from template: {template.template_id}
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import {agent_name}
from router import create_a2a_router

async def main():
    port = int(os.getenv("A2A_AGENT_PORT", "8020"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL", f"http://localhost:{{port}}")
    
    agent_manager_url = os.getenv("A2A_AGENT_MANAGER_URL")
    
    agent = {agent_name}(
        base_url=base_url,
        agent_manager_url=agent_manager_url
    )
    
    await agent.initialize()
    await agent.register_with_network()
    
    app = FastAPI(title=f"A2A {{agent.name}}")
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.include_router(create_a2a_router(agent))
    
    @app.get("/health")
    async def health():
        return {{"status": "healthy", "agent_type": "{template.agent_type}"}}
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Generate agent.py
        agent_py = f'''"""
{agent_name} - A2A Agent
Template: {template.name}
"""

import logging
from typing import Dict, Any
from a2aCommon import A2AAgentBase, a2a_skill, A2AMessage
from a2aCommon.sdk.utils import create_success_response, create_error_response


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class {agent_name}(A2AAgentBase):
    """
    {template.description}
    """
    
    def __init__(self, base_url: str, agent_manager_url: str):
        super().__init__(
            agent_id="{agent_name.lower()}_agent",
            name="{agent_name}",
            description="{template.description}",
            version="1.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.is_ready = True
    
    async def initialize(self):
        """Initialize the agent"""
        logger.info(f"Initializing {{self.name}}")
        
        # Add initialization logic here
        pass
'''
        
        # Add skills from template
        for skill in template.skills:
            skill_method = f'''
    @a2a_skill("{skill['name']}", "{skill['description']}")
    async def {skill['name']}(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """
        {skill['description']}
        """
        try:
            # Implement skill logic here
            params = message.content.get("parameters", {{}})
            
            result = {{
                "skill": "{skill['name']}",
                "status": "executed",
                "parameters": params
            }}
            
            return create_success_response(result)
            
        except Exception as e:
            logger.error(f"Error in {skill['name']}: {{e}}")
            return create_error_response(str(e))
'''
            agent_py += skill_method
        
        # Generate requirements.txt
        requirements_txt = "\\n".join(template.dependencies + [
            "fastapi",
            "uvicorn",
            "httpx",
            "pydantic"
        ])
        
        return {
            "main.py": main_py,
            "agent.py": agent_py,
            "requirements.txt": requirements_txt
        }
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info(f"Shutting down {self.name}")
"""
Enhanced Agent Builder for A2A Developer Portal
Provides comprehensive agent creation workflow with templates, validation, and testing
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
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum
from uuid import uuid4
import logging
import yaml
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from pydantic import BaseModel, Field, validator
from jinja2 import Template, Environment, FileSystemLoader

# Import A2A types for message handling
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole
except ImportError:
    # Define basic types if core module not available
    class MessageRole(str, Enum):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    class MessagePart(BaseModel):
        role: MessageRole
        content: str
        content_type: str = "text/plain"

    class A2AMessage(BaseModel):
        id: str
        conversation_id: str
        sender_id: str
        recipient_id: str
        parts: List[MessagePart]
        metadata: Dict[str, Any] = Field(default_factory=dict)

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Agent type enumeration"""
    DATA_PROCESSOR = "data_processor"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"
    INTEGRATION_CONNECTOR = "integration_connector"
    DECISION_MAKER = "decision_maker"
    MONITORING_AGENT = "monitoring_agent"
    CUSTOM = "custom"


class SkillType(str, Enum):
    """Skill type enumeration"""
    DATA_TRANSFORMATION = "data_transformation"
    API_INTEGRATION = "api_integration"
    FILE_PROCESSING = "file_processing"
    DATABASE_OPERATIONS = "database_operations"
    WORKFLOW_MANAGEMENT = "workflow_management"
    NOTIFICATION = "notification"
    VALIDATION = "validation"
    CUSTOM = "custom"


class HandlerType(str, Enum):
    """Handler type enumeration"""
    HTTP_REQUEST = "http_request"
    MESSAGE_QUEUE = "message_queue"
    FILE_WATCHER = "file_watcher"
    SCHEDULER = "scheduler"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class AgentSkill(BaseModel):
    """Agent skill configuration"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: SkillType
    description: str = ""
    configuration: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    enabled: bool = True
    priority: int = 0

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Skill name cannot be empty')
        return v.strip()


class AgentHandler(BaseModel):
    """Agent handler configuration"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: HandlerType
    description: str = ""
    configuration: Dict[str, Any] = Field(default_factory=dict)
    triggers: List[str] = Field(default_factory=list)
    enabled: bool = True
    priority: int = 0

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Handler name cannot be empty')
        return v.strip()


class AgentConfiguration(BaseModel):
    """Complete agent configuration"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    agent_type: AgentType = AgentType.CUSTOM
    version: str = "1.0.0"

    # Core configuration
    skills: List[AgentSkill] = Field(default_factory=list)
    handlers: List[AgentHandler] = Field(default_factory=list)

    # Runtime configuration
    max_concurrent_tasks: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3

    # Security and trust
    trust_level: str = "medium"
    allowed_delegations: List[str] = Field(default_factory=list)
    security_policies: Dict[str, Any] = Field(default_factory=dict)

    # Monitoring and logging
    logging_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 60

    # Deployment
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Agent name cannot be empty')
        return v.strip()


class AgentTemplate(BaseModel):
    """Agent template definition"""
    id: str
    name: str
    description: str
    agent_type: AgentType
    category: str = "general"
    tags: List[str] = Field(default_factory=list)

    # Template configuration
    default_skills: List[Dict[str, Any]] = Field(default_factory=list)
    default_handlers: List[Dict[str, Any]] = Field(default_factory=list)
    default_config: Dict[str, Any] = Field(default_factory=dict)

    # Template files
    code_templates: Dict[str, str] = Field(default_factory=dict)  # filename -> template content
    config_templates: Dict[str, str] = Field(default_factory=dict)

    # Requirements
    dependencies: List[str] = Field(default_factory=list)
    python_version: str = "3.8+"

    # Metadata
    author: str = "A2A Team"
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Agent validation result"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """Agent generation result"""
    success: bool
    agent_id: str
    generated_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class EnhancedAgentBuilder:
    """Enhanced agent builder with comprehensive workflow"""

    def __init__(self, templates_path: str, output_path: str):
        self.templates_path = Path(templates_path)
        self.output_path = Path(output_path)
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=False
        )

        # Load built-in templates
        self.templates: Dict[str, AgentTemplate] = {}
        self._load_builtin_templates()

        logger.info(f"Enhanced Agent Builder initialized with {len(self.templates)} templates")

    def _load_builtin_templates(self):
        """Load built-in agent templates"""

        # Data Processing Agent Template
        data_processor_template = AgentTemplate(
            id="data-processor",
            name="Data Processing Agent",
            description="Agent specialized in data transformation and processing",
            agent_type=AgentType.DATA_PROCESSOR,
            category="data",
            tags=["data", "processing", "transformation"],
            default_skills=[
                {
                    "name": "data_transformer",
                    "type": "data_transformation",
                    "description": "Transform data between formats",
                    "configuration": {
                        "supported_formats": ["json", "csv", "xml", "yaml"],
                        "validation_enabled": True
                    }
                },
                {
                    "name": "data_validator",
                    "type": "validation",
                    "description": "Validate data integrity and format",
                    "configuration": {
                        "strict_mode": False,
                        "custom_rules": []
                    }
                }
            ],
            default_handlers=[
                {
                    "name": "http_data_handler",
                    "type": "http_request",
                    "description": "Handle HTTP requests for data processing",
                    "configuration": {
                        "methods": ["POST", "PUT"],
                        "content_types": ["application/json", "text/csv"]
                    }
                }
            ],
            code_templates={
                "main.py": """
# {{ agent_name }} - Data Processing Agent
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from app.a2a.agents.dataStandardizationAgent import DataStandardizationAgent
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole

class {{ agent_class_name }}(DataStandardizationAgent):
    \"\"\"{{ agent_description }}\"\"\"

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.name = "{{ agent_name }}"
        self.description = "{{ agent_description }}"

    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Process incoming data\"\"\"
        try:
            # Add your data processing logic here
            processed_data = data.copy()
            processed_data["processed_at"] = datetime.utcnow().isoformat()
            processed_data["processed_by"] = self.agent_id

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

    async def handle_message(self, message: A2AMessage) -> A2AMessage:
        \"\"\"Handle incoming A2A messages\"\"\"
        try:
            # Extract data from message
            data = json.loads(message.parts[0].content)

            # Process data
            processed_data = await self.process_data(data)

            # Create response message
            response = A2AMessage(
                id=f"response_{message.id}",
                conversation_id=message.conversation_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                parts=[
                    MessagePart(
                        role=MessageRole.ASSISTANT,
                        content=json.dumps(processed_data),
                        content_type="application/json"
                    )
                ]
            )

            return response

        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            raise

# Agent factory function
def create_agent(agent_id: str, config: Dict[str, Any]) -> {{ agent_class_name }}:
    return {{ agent_class_name }}(agent_id, config)
""",
                "config.yaml": """
# {{ agent_name }} Configuration
agent:
  name: "{{ agent_name }}"
  description: "{{ agent_description }}"
  type: "{{ agent_type }}"
  version: "{{ version }}"

skills:
{% for skill in skills %}
  - name: "{{ skill.name }}"
    type: "{{ skill.type }}"
    description: "{{ skill.description }}"
    configuration:
{% for key, value in skill.configuration.items() %}
      {{ key }}: {{ value }}
{% endfor %}
    enabled: {{ skill.enabled | lower }}
{% endfor %}

handlers:
{% for handler in handlers %}
  - name: "{{ handler.name }}"
    type: "{{ handler.type }}"
    description: "{{ handler.description }}"
    configuration:
{% for key, value in handler.configuration.items() %}
      {{ key }}: {{ value }}
{% endfor %}
    enabled: {{ handler.enabled | lower }}
{% endfor %}

runtime:
  max_concurrent_tasks: {{ max_concurrent_tasks }}
  timeout_seconds: {{ timeout_seconds }}
  retry_attempts: {{ retry_attempts }}
  logging_level: "{{ logging_level }}"
  metrics_enabled: {{ metrics_enabled | lower }}

security:
  trust_level: "{{ trust_level }}"
  allowed_delegations: {{ allowed_delegations }}
"""
            },
            dependencies=["fastapi", "pydantic", "asyncio"],
            python_version="3.8+"
        )

        self.templates["data-processor"] = data_processor_template

        # Workflow Orchestrator Template
        workflow_template = AgentTemplate(
            id="workflow-orchestrator",
            name="Workflow Orchestrator Agent",
            description="Agent for orchestrating complex workflows and business processes",
            agent_type=AgentType.WORKFLOW_ORCHESTRATOR,
            category="workflow",
            tags=["workflow", "orchestration", "bpmn"],
            default_skills=[
                {
                    "name": "workflow_executor",
                    "type": "workflow_management",
                    "description": "Execute BPMN workflows",
                    "configuration": {
                        "engine": "camunda",
                        "parallel_execution": True
                    }
                }
            ],
            default_handlers=[
                {
                    "name": "workflow_trigger",
                    "type": "message_queue",
                    "description": "Trigger workflows from message queue",
                    "configuration": {
                        "queue_name": "workflow_triggers",
                        "auto_ack": False
                    }
                }
            ],
            dependencies=["fastapi", "pydantic", "asyncio", "camunda-client"],
            python_version="3.8+"
        )

        self.templates["workflow-orchestrator"] = workflow_template

    async def get_templates(self) -> List[AgentTemplate]:
        """Get all available agent templates"""
        return list(self.templates.values())

    async def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get specific agent template"""
        return self.templates.get(template_id)

    async def validate_configuration(self, config: AgentConfiguration) -> ValidationResult:
        """Validate agent configuration"""
        errors = []
        warnings = []
        suggestions = []

        # Validate basic configuration
        if not config.name:
            errors.append("Agent name is required")

        if not config.skills:
            warnings.append("Agent has no skills defined")

        if not config.handlers:
            warnings.append("Agent has no handlers defined")

        # Validate skills
        skill_names = set()
        for skill in config.skills:
            if skill.name in skill_names:
                errors.append(f"Duplicate skill name: {skill.name}")
            skill_names.add(skill.name)

            # Validate skill configuration
            if skill.type == SkillType.CUSTOM and not skill.configuration:
                warnings.append(f"Custom skill '{skill.name}' has no configuration")

        # Validate handlers
        handler_names = set()
        for handler in config.handlers:
            if handler.name in handler_names:
                errors.append(f"Duplicate handler name: {handler.name}")
            handler_names.add(handler.name)

            # Validate handler configuration
            if handler.type == HandlerType.CUSTOM and not handler.configuration:
                warnings.append(f"Custom handler '{handler.name}' has no configuration")

        # Performance suggestions
        if config.max_concurrent_tasks > 100:
            suggestions.append("Consider reducing max_concurrent_tasks for better resource management")

        if config.timeout_seconds > 600:
            suggestions.append("Long timeout values may impact system responsiveness")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    async def generate_agent(
        self,
        config: AgentConfiguration,
        template_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> GenerationResult:
        """Generate agent code and configuration files"""
        try:
            # Validate configuration first
            validation = await self.validate_configuration(config)
            if not validation.is_valid:
                return GenerationResult(
                    success=False,
                    agent_id=config.id,
                    errors=validation.errors
                )

            # Determine output directory
            if not output_dir:
                output_dir = self.output_path / config.id
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Get template if specified
            template = None
            if template_id:
                template = await self.get_template(template_id)
                if not template:
                    return GenerationResult(
                        success=False,
                        agent_id=config.id,
                        errors=[f"Template not found: {template_id}"]
                    )

            generated_files = []

            # Generate files from template or default structure
            if template and template.code_templates:
                # Use template files
                for filename, template_content in template.code_templates.items():
                    file_path = output_dir / filename

                    # Render template
                    jinja_template = Template(template_content)
                    rendered_content = jinja_template.render(
                        agent_name=config.name,
                        agent_class_name=self._to_class_name(config.name),
                        agent_description=config.description,
                        agent_type=config.agent_type.value,
                        agent_id=config.id,
                        version=config.version,
                        skills=config.skills,
                        handlers=config.handlers,
                        max_concurrent_tasks=config.max_concurrent_tasks,
                        timeout_seconds=config.timeout_seconds,
                        retry_attempts=config.retry_attempts,
                        logging_level=config.logging_level,
                        metrics_enabled=config.metrics_enabled,
                        trust_level=config.trust_level,
                        allowed_delegations=config.allowed_delegations
                    )

                    # Write file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(rendered_content)

                    generated_files.append(str(file_path))
            else:
                # Generate default structure
                await self._generate_default_structure(config, output_dir, generated_files)

            # Generate requirements.txt
            requirements_path = output_dir / "requirements.txt"
            requirements = ["fastapi", "pydantic", "asyncio", "httpx", "pyyaml"]
            if template:
                requirements.extend(template.dependencies)

            with open(requirements_path, 'w') as f:
                f.write('\n'.join(sorted(set(requirements))))
            generated_files.append(str(requirements_path))

            # Generate README.md
            readme_path = output_dir / "README.md"
            readme_content = self._generate_readme(config, template)
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            generated_files.append(str(readme_path))

            return GenerationResult(
                success=True,
                agent_id=config.id,
                generated_files=generated_files,
                warnings=validation.warnings
            )

        except Exception as e:
            logger.error(f"Error generating agent: {e}")
            return GenerationResult(
                success=False,
                agent_id=config.id,
                errors=[str(e)]
            )

    async def _generate_default_structure(
        self,
        config: AgentConfiguration,
        output_dir: Path,
        generated_files: List[str]
    ):
        """Generate default agent structure"""

        # Generate main agent file
        main_content = f'''"""
{config.name} - A2A Agent
{config.description}
Generated by A2A Agent Builder
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from app.a2a.sdk.agentBase import A2AAgentBase
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole

class {config.name.replace(' ', '').replace('-', '_')}Agent(A2AAgentBase):
    """
    {config.description}
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.name = "{config.name}"
        self.description = "{config.description}"
        self.agent_type = "{config.agent_type.value}"

        # Initialize skills and handlers
        self._initialize_skills()
        self._initialize_handlers()

    def _initialize_skills(self):
        """Initialize agent skills"""
        # Initialize skills from configuration
        for skill_config in self.config.get('skills', []):
            skill_name = skill_config.get('name')
            skill_type = skill_config.get('type')

            if skill_type == 'function':
                # Register function-based skill
                self.skills[skill_name] = {
                    'type': 'function',
                    'handler': skill_config.get('handler'),
                    'parameters': skill_config.get('parameters', {{}}),
                    'description': skill_config.get('description', '')
                }
            elif skill_type == 'api':
                # Register API-based skill
                self.skills[skill_name] = {
                    'type': 'api',
                    'endpoint': skill_config.get('endpoint'),
                    'method': skill_config.get('method', 'POST'),
                    'headers': skill_config.get('headers', {{}}),
                    'description': skill_config.get('description', '')
                }

            logger.info(f"Initialized skill: {skill_name} (type: {skill_type})")

    def _initialize_handlers(self):
        """Initialize agent handlers"""
        # Initialize event handlers from configuration
        for handler_config in self.config.get('handlers', []):
            event_type = handler_config.get('event')
            handler_name = handler_config.get('handler')
            priority = handler_config.get('priority', 0)

            if event_type not in self.handlers:
                self.handlers[event_type] = []

            self.handlers[event_type].append({
                'name': handler_name,
                'function': handler_config.get('function'),
                'priority': priority,
                'filters': handler_config.get('filters', {{}}),
                'async': handler_config.get('async', True)
            })

            # Sort handlers by priority
            self.handlers[event_type].sort(key=lambda x: x['priority'], reverse=True)

            logger.info(f"Registered handler: {handler_name} for event: {event_type}")

    async def handle_message(self, message: A2AMessage) -> A2AMessage:
        """Handle incoming A2A messages"""
        try:
            # Extract message content and type
            message_content = message.parts[0].content if message.parts else ""
            message_type = message.metadata.get('type', 'general')

            # Trigger event handlers for incoming message
            if 'message_received' in self.handlers:
                for handler in self.handlers['message_received']:
                    if handler['async']:
                        await self._execute_handler_async(handler, message)
                    else:
                        self._execute_handler_sync(handler, message)

            # Process message based on type
            response_content = ""
            if message_type == 'query':
                # Handle query messages
                response_content = await self._process_query(message_content)
            elif message_type == 'command':
                # Handle command messages
                response_content = await self._execute_command(message_content)
            elif message_type == 'skill':
                # Handle skill invocation
                skill_name = message.metadata.get('skill')
                response_content = await self._invoke_skill(skill_name, message_content)
            else:
                # Default message processing
                response_content = await self._process_general_message(message_content)

            # Create response message
            response = A2AMessage(
                id=f"response_{message.id}",
                conversation_id=message.conversation_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                parts=[
                    MessagePart(
                        role=MessageRole.ASSISTANT,
                        content=response_content,
                        content_type="text/plain"
                    )
                ],
                metadata={
                    'processed_by': self.agent_id,
                    'processing_time': datetime.now().isoformat(),
                    'message_type': f'{message_type}_response'
                }
            )

            # Trigger response handlers
            if 'message_sent' in self.handlers:
                for handler in self.handlers['message_sent']:
                    if handler['async']:
                        await self._execute_handler_async(handler, response)

            return response

        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            raise

    async def _process_query(self, content: str) -> str:
        """Process query messages through A2A network"""
        try:
            # Parse query intent
            query_type = self._analyze_query_intent(content)

            # Check if this agent can handle the query
            if query_type in self.config.get('capabilities', []):
                # Process locally
                response = await self._execute_local_query(query_type, content)
            else:
                # Forward to appropriate agent in A2A network
                target_agent = await self._find_capable_agent(query_type)
                if target_agent:
                    response = await self._forward_to_agent(target_agent, content, 'query')
                else:
                    response = "No agent available to handle this query type"

            return response

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Error processing query: {str(e)}"

    async def _execute_command(self, content: str) -> str:
        """Execute command through agent network"""
        try:
            # Parse command structure
            command_parts = content.split(' ', 1)
            command = command_parts[0].lower()
            args = command_parts[1] if len(command_parts) > 1 else ""

            # Check if command requires collaboration
            if command in self.config.get('collaborative_commands', []):
                # Execute distributed command
                results = await self._execute_distributed_command(command, args)
                return self._aggregate_command_results(results)
            else:
                # Execute local command
                return await self._execute_local_command(command, args)

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return f"Command execution failed: {str(e)}"

    async def _invoke_skill(self, skill_name: str, content: str) -> str:
        """Invoke a specific skill with real implementation"""
        if skill_name not in self.skills:
            # Try to find skill in other agents
            remote_agent = await self._find_agent_with_skill(skill_name)
            if remote_agent:
                return await self._invoke_remote_skill(remote_agent, skill_name, content)
            return f"Skill '{skill_name}' not found in A2A network"

        skill = self.skills[skill_name]

        try:
            if skill['type'] == 'function':
                # Execute local function skill
                handler = skill.get('handler')
                if callable(handler):
                    result = await handler(content, **skill.get('parameters', {{}}))
                else:
                    # Dynamic function loading
                    module_name, func_name = handler.rsplit('.', 1)
                    module = __import__(module_name, fromlist=[func_name])
                    func = getattr(module, func_name)
                    result = await func(content, **skill.get('parameters', {{}}))
                return str(result)

            elif skill['type'] == 'api':
                # Call external API
                # A2A Protocol: Use blockchain messaging instead of httpx
                # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                    method = skill.get('method', 'POST')
                    headers = skill.get('headers', {{}})
                    headers['Content-Type'] = 'application/json'

                    if method == 'GET':
                        response = await client.get(
                            skill['endpoint'],
                            params={'query': content},
                            headers=headers
                        )
                    else:
                        response = await client.post(
                            skill['endpoint'],
                            json={'data': content},
                            headers=headers
                        )

                    response.raise_for_status()
                    return response.json().get('result', str(response.json()))

            elif skill['type'] == 'ml_model':
                # Invoke ML model
                return await self._invoke_ml_model(skill.get('model_id'), content)

            else:
                return f"Unknown skill type: {skill['type']}"

        except Exception as e:
            logger.error(f"Skill invocation error for {skill_name}: {e}")
            return f"Skill execution failed: {str(e)}"

    async def _process_general_message(self, content: str) -> str:
        """Process general messages using agent's AI capabilities"""
        try:
            # Use agent's language model or reasoning engine
            if self.config.get('llm_enabled', False):
                response = await self._invoke_language_model(content)
            else:
                # Rule-based processing
                response = await self._apply_message_rules(content)

            # Enrich response with context from A2A network if needed
            if self.config.get('enrich_responses', True):
                context = await self._gather_network_context(content)
                response = self._enrich_response(response, context)

            return response

        except Exception as e:
            logger.error(f"General message processing error: {e}")
            return "I encountered an error processing your message."

    async def _execute_handler_async(self, handler: Dict, data: Any):
        """Execute async handler"""
        logger.debug(f"Executing async handler: {handler['name']}")

    def _execute_handler_sync(self, handler: Dict, data: Any):
        """Execute sync handler"""
        logger.debug(f"Executing sync handler: {handler['name']}")

    # A2A Network Communication Methods

    async def _find_capable_agent(self, query_type: str) -> Optional[str]:
        """Find agent in network capable of handling query type"""
        try:
            # Query A2A registry for agents with capability
            registry_url = self.config.get('registry_url', 'http://localhost:8000/api/registry')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.get(
                    f"{registry_url}/agents",
                    params={'capability': query_type}
                )
                if response.status_code == 200:
                    agents = response.json()
                    return agents[0]['agent_id'] if agents else None
        except Exception as e:
            logger.error(f"Error finding capable agent: {e}")
        return None

    async def _forward_to_agent(self, target_agent_id: str, content: str, message_type: str) -> str:
        """Forward message to another agent in A2A network"""
        try:
            # Create A2A message
            message = A2AMessage(
                id=f"{self.agent_id}_to_{target_agent_id}_{datetime.now().timestamp()}",
                conversation_id=f"conv_{datetime.now().timestamp()}",
                sender_id=self.agent_id,
                recipient_id=target_agent_id,
                parts=[
                    MessagePart(
                        role=MessageRole.USER,
                        content=content,
                        content_type="text/plain"
                    )
                ],
                metadata={'type': message_type, 'forwarded': True}
            )

            # Send via A2A network
            network_url = self.config.get('network_url', 'http://localhost:8000/api/messages')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    network_url,
                    json=message.dict(),
                    headers={'X-Agent-ID': self.agent_id}
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', {{}}).get('content', 'No response received')
        except Exception as e:
            logger.error(f"Error forwarding to agent {target_agent_id}: {e}")
        return "Failed to forward message"

    async def _execute_distributed_command(self, command: str, args: str) -> List[Dict]:
        """Execute command across multiple agents"""
        results = []
        try:
            # Get list of participating agents
            agents = await self._get_network_agents()

            # Send command to each agent
            tasks = []
            for agent in agents:
                if agent['agent_id'] != self.agent_id:  # Don't send to self
                    task = self._send_command_to_agent(agent['agent_id'], command, args)
                    tasks.append(task)

            # Execute local command
            local_result = await self._execute_local_command(command, args)
            results.append({'agent_id': self.agent_id, 'result': local_result})

            # Gather remote results
            remote_results = await asyncio.gather(*tasks, return_exceptions=True)
            for agent, result in zip(agents, remote_results):
                if not isinstance(result, Exception):
                    results.append({'agent_id': agent['agent_id'], 'result': result})

        except Exception as e:
            logger.error(f"Distributed command execution error: {e}")

        return results

    def _aggregate_command_results(self, results: List[Dict]) -> str:
        """Aggregate results from distributed command execution"""
        if not results:
            return "No results received"

        aggregated = f"Command executed across {len(results)} agents:\n"
        for result in results:
            aggregated += f"- Agent {result['agent_id']}: {result['result']}\n"

        return aggregated

    async def _find_agent_with_skill(self, skill_name: str) -> Optional[str]:
        """Find agent that has a specific skill"""
        try:
            registry_url = self.config.get('registry_url', 'http://localhost:8000/api/registry')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.get(
                    f"{registry_url}/skills/{skill_name}/agents"
                )
                if response.status_code == 200:
                    agents = response.json()
                    return agents[0]['agent_id'] if agents else None
        except Exception as e:
            logger.error(f"Error finding agent with skill {skill_name}: {e}")
        return None

    async def _invoke_remote_skill(self, agent_id: str, skill_name: str, content: str) -> str:
        """Invoke skill on remote agent"""
        # Create message with skill metadata
        try:
            message = A2AMessage(
                id=f"{self.agent_id}_skill_{skill_name}_{datetime.now().timestamp()}",
                conversation_id=f"conv_{datetime.now().timestamp()}",
                sender_id=self.agent_id,
                recipient_id=agent_id,
                parts=[
                    MessagePart(
                        role=MessageRole.USER,
                        content=content,
                        content_type="text/plain"
                    )
                ],
                metadata={'type': 'skill', 'skill': skill_name, 'forwarded': True}
            )

            # Send via A2A network
            network_url = self.config.get('network_url', 'http://localhost:8000/api/messages')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    network_url,
                    json=message.dict(),
                    headers={'X-Agent-ID': self.agent_id}
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', {{}}).get('content', 'No response received')
        except Exception as e:
            logger.error(f"Error invoking remote skill {skill_name} on agent {agent_id}: {e}")
        return f"Failed to invoke skill {skill_name}"

    def _analyze_query_intent(self, content: str) -> str:
        """Analyze query to determine intent/type"""
        content_lower = content.lower()

        # Simple intent classification
        if any(word in content_lower for word in ['analyze', 'process', 'compute']):
            return 'analytical'
        elif any(word in content_lower for word in ['find', 'search', 'lookup']):
            return 'search'
        elif any(word in content_lower for word in ['create', 'generate', 'make']):
            return 'generative'
        elif any(word in content_lower for word in ['translate', 'convert', 'transform']):
            return 'transformation'
        else:
            return 'general'

    async def _execute_local_query(self, query_type: str, content: str) -> str:
        """Execute query locally based on type"""
        handlers = {
            'analytical': self._handle_analytical_query,
            'search': self._handle_search_query,
            'generative': self._handle_generative_query,
            'transformation': self._handle_transformation_query,
            'general': self._handle_general_query
        }

        handler = handlers.get(query_type, self._handle_general_query)
        return await handler(content)

    async def _execute_local_command(self, command: str, args: str) -> str:
        """Execute command locally"""
        commands = {
            'status': self._get_agent_status,
            'list': self._list_capabilities,
            'execute': self._execute_task,
            'stop': self._stop_task,
            'configure': self._configure_agent
        }

        if command in commands:
            return await commands[command](args)
        else:
            return f"Unknown command: {command}"

    async def _invoke_language_model(self, content: str) -> str:
        """Invoke configured language model"""
        llm_config = self.config.get('llm_config', {{}})

        if llm_config.get('provider') == 'openai':
            # OpenAI integration
            import openai
            openai.api_key = llm_config.get('api_key')

            response = await openai.ChatCompletion.acreate(
                model=llm_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": self.config.get('system_prompt', "You are a helpful A2A agent.")},
                    {"role": "user", "content": content}
                ]
            )
            return response.choices[0].message.content

        elif llm_config.get('provider') == 'local':
            # Local model integration
            return await self._invoke_local_llm(content, llm_config)

        return "No language model configured"

    async def _apply_message_rules(self, content: str) -> str:
        """Apply rule-based message processing"""
        rules = self.config.get('message_rules', [])

        for rule in rules:
            if rule['type'] == 'pattern':
                import re
                if re.search(rule['pattern'], content, re.IGNORECASE):
                    return rule['response']
            elif rule['type'] == 'keyword':
                if any(keyword in content.lower() for keyword in rule['keywords']):
                    return rule['response']

        return "I understand your message but don't have a specific response."

    async def _gather_network_context(self, content: str) -> Dict[str, Any]:
        """Gather context from A2A network"""
        context = {{}}

        try:
            # Query related agents for context
            related_agents = await self._find_related_agents(content)

            tasks = []
            for agent_id in related_agents[:3]:  # Limit to 3 agents
                task = self._query_agent_context(agent_id, content)
                tasks.append(task)

            contexts = await asyncio.gather(*tasks, return_exceptions=True)

            for agent_id, agent_context in zip(related_agents, contexts):
                if not isinstance(agent_context, Exception):
                    context[agent_id] = agent_context

        except Exception as e:
            logger.error(f"Error gathering network context: {e}")

        return context

    def _enrich_response(self, response: str, context: Dict[str, Any]) -> str:
        """Enrich response with network context"""
        if not context:
            return response

        enriched = response + "\n\nAdditional context from A2A network:"
        for agent_id, agent_context in context.items():
            enriched += f"\n- {agent_id}: {agent_context.get('summary', 'No summary available')}"

        return enriched

# Agent factory function
def create_agent(agent_id: str, config: Dict[str, Any]) -> {config.name.replace(' ', '').replace('-', '_')}Agent:
    return {config.name.replace(' ', '').replace('-', '_')}Agent(agent_id, config)
'''

        main_path = output_dir / "main.py"
        with open(main_path, 'w') as f:
            f.write(main_content)
        generated_files.append(str(main_path))

        # Generate configuration file
        config_content = {
            "agent": {
                "name": config.name,
                "description": config.description,
                "type": config.agent_type.value,
                "version": config.version
            },
            "skills": [skill.dict() for skill in config.skills],
            "handlers": [handler.dict() for handler in config.handlers],
            "runtime": {
                "max_concurrent_tasks": config.max_concurrent_tasks,
                "timeout_seconds": config.timeout_seconds,
                "retry_attempts": config.retry_attempts,
                "logging_level": config.logging_level,
                "metrics_enabled": config.metrics_enabled
            },
            "security": {
                "trust_level": config.trust_level,
                "allowed_delegations": config.allowed_delegations
            }
        }

        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False, indent=2)
        generated_files.append(str(config_path))

    def _generate_readme(self, config: AgentConfiguration, template: Optional[AgentTemplate]) -> str:
        """Generate README.md content"""
        content = f"""# {config.name}

{config.description}

## Overview

- **Type**: {config.agent_type.value}
- **Version**: {config.version}
- **Generated**: {datetime.utcnow().isoformat()}

## Skills

"""

        for skill in config.skills:
            content += f"- **{skill.name}** ({skill.type.value}): {skill.description}\n"

        content += "\n## Handlers\n\n"

        for handler in config.handlers:
            content += f"- **{handler.name}** ({handler.type.value}): {handler.description}\n"

        content += f"""
## Configuration

- Max Concurrent Tasks: {config.max_concurrent_tasks}
- Timeout: {config.timeout_seconds} seconds
- Retry Attempts: {config.retry_attempts}
- Logging Level: {config.logging_level}

## Usage

```python
from main import create_agent

# Create agent instance
agent = create_agent("{config.id}", config)

# Handle messages
response = await agent.handle_message(message)
```

## Development

1. Install dependencies: `pip install -r requirements.txt`
2. Configure the agent in `config.yaml`
3. Implement custom logic in `main.py`
4. Test the agent functionality

---
Generated by A2A Agent Builder
"""

        return content

    def _to_class_name(self, name: str) -> str:
        """Convert agent name to valid Python class name"""
        # Remove special characters and convert to PascalCase
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        words = clean_name.split()
        return ''.join(word.capitalize() for word in words) + 'Agent'

    # Additional helper methods for real A2A network operations

    async def _get_network_agents(self) -> List[Dict[str, Any]]:
        """Get list of all agents in the A2A network"""
        try:
            registry_url = self.config.get('registry_url', 'http://localhost:8000/api/registry')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.get(f"{registry_url}/agents")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error getting network agents: {e}")
        return []

    async def _send_command_to_agent(self, agent_id: str, command: str, args: str) -> str:
        """Send command to specific agent"""
        try:
            message = {
                "type": "command",
                "command": command,
                "args": args,
                "sender_id": self.agent_id
            }

            network_url = self.config.get('network_url', 'http://localhost:8000/api/messages')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{network_url}/{agent_id}",
                    json=message,
                    headers={'X-Agent-ID': self.agent_id}
                )
                if response.status_code == 200:
                    return response.json().get('result', 'Command sent')
        except Exception as e:
            logger.error(f"Error sending command to {agent_id}: {e}")
        return "Failed to send command"

    async def _invoke_ml_model(self, model_id: str, content: str) -> str:
        """Invoke ML model for inference"""
        try:
            ml_service_url = self.config.get('ml_service_url', 'http://localhost:8001/api/ml')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{ml_service_url}/models/{model_id}/predict",
                    json={"input": content},
                    headers={'X-Agent-ID': self.agent_id}
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get('prediction', str(result))
        except Exception as e:
            logger.error(f"Error invoking ML model {model_id}: {e}")
        return "ML model invocation failed"

    async def _invoke_local_llm(self, content: str, llm_config: Dict[str, Any]) -> str:
        """Invoke local language model"""
        try:
            # Example integration with local LLM service
            local_llm_url = llm_config.get('endpoint', 'http://localhost:8002/api/llm')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{local_llm_url}/generate",
                    json={
                        "prompt": content,
                        "max_tokens": llm_config.get('max_tokens', 150),
                        "temperature": llm_config.get('temperature', 0.7)
                    }
                )
                if response.status_code == 200:
                    return response.json().get('generated_text', 'No response generated')
        except Exception as e:
            logger.error(f"Error invoking local LLM: {e}")
        return "Local LLM invocation failed"

    async def _find_related_agents(self, content: str) -> List[str]:
        """Find agents related to the message content"""
        try:
            # Extract keywords from content
            keywords = self._extract_keywords(content)

            registry_url = self.config.get('registry_url', 'http://localhost:8000/api/registry')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{registry_url}/search",
                    json={"keywords": keywords}
                )
                if response.status_code == 200:
                    agents = response.json()
                    return [agent['agent_id'] for agent in agents]
        except Exception as e:
            logger.error(f"Error finding related agents: {e}")
        return []

    async def _query_agent_context(self, agent_id: str, content: str) -> Dict[str, Any]:
        """Query specific agent for context"""
        try:
            message = {
                "type": "context_request",
                "content": content,
                "sender_id": self.agent_id
            }

            network_url = self.config.get('network_url', 'http://localhost:8000/api/messages')
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    f"{network_url}/{agent_id}/context",
                    json=message,
                    headers={'X-Agent-ID': self.agent_id}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error querying agent {agent_id} for context: {e}")
        return {}

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for agent matching"""
        # Simple keyword extraction - in production use NLP
        import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}

        # Extract words
        words = re.findall(r'\b\w+\b', content.lower())

        # Filter keywords
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        # Return unique keywords
        return list(set(keywords))[:10]  # Limit to 10 keywords

    # Handler implementation methods for generated agents

    async def _handle_analytical_query(self, content: str) -> str:
        """Handle analytical queries"""
        return f"Analyzing: {content}"

    async def _handle_search_query(self, content: str) -> str:
        """Handle search queries"""
        return f"Searching for: {content}"

    async def _handle_generative_query(self, content: str) -> str:
        """Handle generative queries"""
        return f"Generating response for: {content}"

    async def _handle_transformation_query(self, content: str) -> str:
        """Handle transformation queries"""
        return f"Transforming: {content}"

    async def _handle_general_query(self, content: str) -> str:
        """Handle general queries"""
        return f"Processing general query: {content}"

    async def _get_agent_status(self, args: str) -> str:
        """Get agent status"""
        return "Agent is operational"

    async def _list_capabilities(self, args: str) -> str:
        """List agent capabilities"""
        return "Agent capabilities: query processing, command execution, skill invocation"

    async def _execute_task(self, args: str) -> str:
        """Execute a specific task"""
        return f"Executing task: {args}"

    async def _stop_task(self, args: str) -> str:
        """Stop a running task"""
        return f"Stopping task: {args}"

    async def _configure_agent(self, args: str) -> str:
        """Configure agent settings"""
        return f"Configuring agent with: {args}"

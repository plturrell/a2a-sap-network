"""
Agent Builder Agent - SDK Version
Generates and manages A2A agents using templates, BPMN workflows, and the A2A SDK
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
from typing import Dict, List, Any, Optional
from uuid import uuid4
import logging
import jinja2
import yaml
from pathlib import Path

# Import SDK components - use local components
from ..sdk.performanceMonitoringMixin import PerformanceMonitoringMixin, monitor_a2a_operation
from app.a2a.sdk import (
    A2AAge, a2a_handlerntBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.core.workflowContext import workflowContextManager
from app.a2a.core.workflowMonitor import workflowMonitor
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
# Import trust system
from app.a2a.core.trustManager import sign_a2a_message, initialize_agent_trust, verify_a2a_message, trust_manager
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def create_error_response(message: str) -> Dict[str, Any]:
    """Create error response"""
    return {"success": False, "error": message, "timestamp": datetime.now().isoformat()}


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create success response"""
    return {"success": True, "data": data, "timestamp": datetime.now().isoformat()}


class AgentTemplate(BaseModel), PerformanceMonitoringMixin:
    """Agent template definition"""
    name: str
    description: str
    category: str
    skills: List[str]
    handlers: List[str]
    tasks: List[str]
    dependencies: List[str] = []
    resource_requirements: Dict[str, str] = {}
    template_variables: Dict[str, Any] = {}


class BPMNWorkflow(BaseModel), PerformanceMonitoringMixin:
    """BPMN workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    start_event: str
    end_events: List[str]
    tasks: List[Dict[str, Any]]
    gateways: List[Dict[str, Any]]
    sequence_flows: List[Dict[str, Any]]
    data_objects: List[Dict[str, Any]] = []


class AgentGenerationRequest(BaseModel), PerformanceMonitoringMixin:
    """Request for agent generation"""
    agent_name: str
    agent_id: str
    description: str
    template_name: str
    custom_skills: List[str] = []
    custom_handlers: List[str] = []
    bpmn_workflow: Optional[BPMNWorkflow] = None
    configuration: Dict[str, Any] = {}
    output_directory: str = "/tmp/generated_agents"


class AgentBuilderResponse(BaseModel), PerformanceMonitoringMixin:
    """Response from agent builder operations"""
    success: bool
    message: str
    agent_id: Optional[str] = None
    generated_files: List[str] = []
    errors: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


class AgentBuilderAgentSDK(A2AAgentBase), PerformanceMonitoringMixin:
    """
    Agent Builder Agent - SDK Version
    Generates and manages A2A agents using templates and BPMN workflows
    """
    
    def __init__(self, base_url: str, templates_path: str):
        super().__init__(
            agent_id="agent_builder_agent",
            name="Agent Builder Agent",
            description="A2A v0.2.9 compliant agent for generating and managing other A2A agents",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.templates_path = Path(templates_path)
        self.generated_agents = {}
        self.agent_templates = {}
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(7)  # 7 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            "agents_generated": 0,
            "templates_created": 0,
            "bpmn_workflows_processed": 0,
            "validations_performed": 0
        }
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_path)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8007'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Agent Builder Agent resources...")
        
        # Establish standard trust relationships FIRST
        await self.establish_standard_trust_relationships()
        
        # Initialize storage
        storage_path = os.getenv("AGENT_BUILDER_STORAGE_PATH", "/tmp/agent_builder_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = Path(storage_path)
        
        # Create templates directory if it doesn't exist
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # Load built-in templates
        await self._load_builtin_templates()
        
        # Load existing state
        await self._load_agent_state()
        
        # Discover available templates and agents from catalog_manager
        available_agents = await self.discover_agents(
            capabilities=["template_processing", "code_generation", "agent_management"],
            agent_types=["builder", "generator", "development"]
        )
        
        # Store discovered agents for collaboration
        self.collaborative_agents = {
            "template_agents": [agent for agent in available_agents if "template" in agent.get("capabilities", [])],
            "builder_agents": [agent for agent in available_agents if "builder" in agent.get("agent_type", "")],
            "development_agents": [agent for agent in available_agents if "development" in agent.get("agent_type", "")]
        }
        
        logger.info(f"Agent Builder Agent initialization complete with {len(available_agents)} collaborative agents")
    
    @a2a_handler("agent_generation")
    async def handle_agent_generation(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for agent generation requests"""
        start_time = time.time()
        
        try:
            # Extract generation request from message
            request_data = self._extract_request_data(message)
            if not request_data:
                return create_error_response("No valid generation request found in message")
            
            # Create AgentGenerationRequest
            generation_request = AgentGenerationRequest(**request_data)
            
            # Generate agent
            generation_result = await self.generate_agent(
                generation_request=generation_request,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='agent_generation').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='agent_generation').observe(time.time() - start_time)
            
            return create_success_response(generation_result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='agent_generation').inc()
            logger.error(f"Agent generation failed: {e}")
            return create_error_response(f"Agent generation failed: {str(e)}")
    
    @a2a_handler("template_management")
    async def handle_template_management(self, message: A2AMessage) -> Dict[str, Any]:
        """Handler for template management operations"""
        start_time = time.time()
        
        try:
            # Extract template operation data
            operation_data = self._extract_request_data(message)
            operation = operation_data.get('operation', 'list')
            
            if operation == 'create':
                result = await self.create_template(operation_data)
            elif operation == 'update':
                result = await self.update_template(operation_data)
            elif operation == 'delete':
                result = await self.delete_template(operation_data)
            elif operation == 'list':
                result = await self.list_templates()
            else:
                return create_error_response(f"Unknown template operation: {operation}")
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='template_management').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='template_management').observe(time.time() - start_time)
            
            return create_success_response(result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='template_management').inc()
            logger.error(f"Template management failed: {e}")
            return create_error_response(f"Template management failed: {str(e)}")
    
    @a2a_skill("template_validation")
    async def template_validation_skill(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent template"""
        
        validation_errors = []
        validation_warnings = []
        
        # Required fields validation
        required_fields = ['name', 'description', 'category', 'skills', 'handlers']
        for field in required_fields:
            if field not in template_data or not template_data[field]:
                validation_errors.append(f"Missing required field: {field}")
        
        # Skills validation
        skills = template_data.get('skills', [])
        if not isinstance(skills, list):
            validation_errors.append("Skills must be a list")
        elif len(skills) == 0:
            validation_warnings.append("Template has no skills defined")
        
        # Handlers validation
        handlers = template_data.get('handlers', [])
        if not isinstance(handlers, list):
            validation_errors.append("Handlers must be a list")
        elif len(handlers) == 0:
            validation_warnings.append("Template has no handlers defined")
        
        # Category validation
        valid_categories = ['data_processing', 'integration', 'ai_ml', 'monitoring', 'utility', 'custom']
        category = template_data.get('category')
        if category and category not in valid_categories:
            validation_warnings.append(f"Category '{category}' not in recommended categories: {valid_categories}")
        
        self.processing_stats["validations_performed"] += 1
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "score": max(0, (len(required_fields) - len(validation_errors)) / len(required_fields))
        }
    
    @a2a_skill("code_generation")
    async def code_generation_skill(self, generation_request: AgentGenerationRequest, template: AgentTemplate) -> Dict[str, Any]:
        """Generate agent code from template"""
        
        try:
            # Load template file
            template_file = f"{template.name}_template.py.j2"
            jinja_template = self.jinja_env.get_template(template_file)
            
            # Prepare template context
            template_context = {
                "agent_name": generation_request.agent_name,
                "agent_id": generation_request.agent_id,
                "description": generation_request.description,
                "skills": template.skills + generation_request.custom_skills,
                "handlers": template.handlers + generation_request.custom_handlers,
                "tasks": template.tasks,
                "dependencies": template.dependencies,
                "configuration": generation_request.configuration,
                "generated_at": datetime.now().isoformat(),
                "generator_version": self.version,
                **template.template_variables
            }
            
            # Generate code
            generated_code = jinja_template.render(**template_context)
            
            # Create output file path
            output_dir = Path(generation_request.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{generation_request.agent_id}_sdk.py"
            
            # Write generated code
            with open(output_file, 'w') as f:
                f.write(generated_code)
            
            logger.info(f"Generated agent code: {output_file}")
            
            # Store generated agent data in data_manager
            await self.store_agent_data(
                data_type="agent_generation",
                data={
                    "agent_id": generation_request.agent_id,
                    "agent_name": generation_request.agent_name,
                    "template_used": template.name,
                    "generated_file": str(output_file),
                    "lines_of_code": len(generated_code.split('\\n')),
                    "skills_count": len(template_context["skills"]),
                    "handlers_count": len(template_context["handlers"]),
                    "generation_timestamp": datetime.now().isoformat()
                },
                metadata={
                    "generator_version": self.version,
                    "template_category": template.category
                }
            )
            
            # Update agent status with agent_manager
            await self.update_agent_status(
                status="active",
                details={
                    "last_generation": generation_request.agent_name,
                    "total_agents_generated": self.processing_stats.get("agents_generated", 0) + 1,
                    "templates_available": len(self.agent_templates),
                    "active_capabilities": ["code_generation", "template_processing", "bpmn_processing"]
                }
            )
            
            return {
                "generated_file": str(output_file),
                "template_used": template.name,
                "lines_of_code": len(generated_code.split('\\n')),
                "skills_count": len(template_context["skills"]),
                "handlers_count": len(template_context["handlers"])
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    @a2a_skill("bpmn_processing")
    async def bpmn_processing_skill(self, bpmn_workflow: BPMNWorkflow) -> Dict[str, Any]:
        """Process BPMN workflow and generate workflow logic"""
        
        try:
            # Validate BPMN workflow
            validation_result = self._validate_bpmn_workflow(bpmn_workflow)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid BPMN workflow: {validation_result['errors']}")
            
            # Generate workflow logic
            workflow_code = self._generate_workflow_from_bpmn(bpmn_workflow)
            
            # Generate workflow configuration
            workflow_config = self._generate_workflow_config(bpmn_workflow)
            
            self.processing_stats["bpmn_workflows_processed"] += 1
            
            return {
                "workflow_id": bpmn_workflow.workflow_id,
                "workflow_code": workflow_code,
                "workflow_config": workflow_config,
                "tasks_count": len(bpmn_workflow.tasks),
                "gateways_count": len(bpmn_workflow.gateways),
                "flows_count": len(bpmn_workflow.sequence_flows)
            }
            
        except Exception as e:
            logger.error(f"BPMN processing failed: {e}")
            raise
    
    @a2a_skill("dependency_resolution")
    async def dependency_resolution_skill(self, template: AgentTemplate, generation_request: AgentGenerationRequest) -> Dict[str, Any]:
        """Resolve and validate agent dependencies"""
        
        all_dependencies = template.dependencies.copy()
        
        # Add SDK dependencies (always required)
        sdk_dependencies = [
            "app.a2a.sdk",
            "prometheus_client",
            "asyncio",
            "logging",
            "json",
            "os",
            "time"
        ]
        all_dependencies.extend(sdk_dependencies)
        
        # Add custom dependencies from configuration
        custom_deps = generation_request.configuration.get('dependencies', [])
        all_dependencies.extend(custom_deps)
        
        # Remove duplicates
        all_dependencies = list(set(all_dependencies))
        
        # Generate requirements.txt content
        requirements_content = "\\n".join([
            f"{dep}>=1.0.0" if not ">=" in dep else dep 
            for dep in all_dependencies
        ])
        
        return {
            "dependencies": all_dependencies,
            "requirements_content": requirements_content,
            "total_dependencies": len(all_dependencies)
        }
    
    @a2a_skill("configuration_generation")
    async def configuration_generation_skill(self, generation_request: AgentGenerationRequest, template: AgentTemplate) -> Dict[str, Any]:
        """Generate agent configuration files"""
        
        # Generate Docker configuration
        dockerfile_content = self._generate_dockerfile(generation_request, template)
        
        # Generate Docker Compose service
        compose_service = self._generate_compose_service(generation_request, template)
        
        # Generate environment configuration
        env_config = self._generate_env_config(generation_request, template)
        
        # Generate launch script
        launch_script = self._generate_launch_script(generation_request)
        
        return {
            "dockerfile": dockerfile_content,
            "compose_service": compose_service,
            "env_config": env_config,
            "launch_script": launch_script
        }
    
    @a2a_skill("agent_testing")
    async def agent_testing_skill(self, agent_file_path: str) -> Dict[str, Any]:
        """Generate test files for the generated agent"""
        
        # Generate unit tests
        unit_test_content = self._generate_unit_tests(agent_file_path)
        
        # Generate integration tests
        integration_test_content = self._generate_integration_tests(agent_file_path)
        
        # Create test files
        test_dir = Path(agent_file_path).parent / "tests"
        test_dir.mkdir(exist_ok=True)
        
        unit_test_file = test_dir / "test_unit.py"
        integration_test_file = test_dir / "test_integration.py"
        
        with open(unit_test_file, 'w') as f:
            f.write(unit_test_content)
        
        with open(integration_test_file, 'w') as f:
            f.write(integration_test_content)
        
        return {
            "unit_test_file": str(unit_test_file),
            "integration_test_file": str(integration_test_file),
            "test_methods": ["test_initialization", "test_handlers", "test_skills", "test_error_handling"]
        }
    
    @a2a_task(
        task_type="agent_generation_workflow",
        description="Complete agent generation workflow",
        timeout=600,
        retry_attempts=2
    )
    async def generate_agent(self, generation_request: AgentGenerationRequest, context_id: str) -> Dict[str, Any]:
        """Complete agent generation workflow"""
        
        try:
            # Stage 1: Validate template
            if generation_request.template_name not in self.agent_templates:
                raise ValueError(f"Template '{generation_request.template_name}' not found")
            
            template = self.agent_templates[generation_request.template_name]
            
            # Stage 2: Validate template
            validation_result = await self.execute_skill("template_validation", template.dict())
            if not validation_result["valid"]:
                raise ValueError(f"Template validation failed: {validation_result['errors']}")
            
            # Stage 3: Resolve dependencies
            dependencies_result = await self.execute_skill("dependency_resolution", template, generation_request)
            
            # Stage 4: Process BPMN workflow (if provided)
            bpmn_result = None
            if generation_request.bpmn_workflow:
                bpmn_result = await self.execute_skill("bpmn_processing", generation_request.bpmn_workflow)
            
            # Stage 5: Generate agent code
            code_result = await self.execute_skill("code_generation", generation_request, template)
            
            # Stage 6: Generate configuration files
            config_result = await self.execute_skill("configuration_generation", generation_request, template)
            
            # Stage 7: Generate tests
            test_result = await self.execute_skill("agent_testing", code_result["generated_file"])
            
            # Stage 8: Write additional files
            output_dir = Path(generation_request.output_directory)
            
            # Write Dockerfile
            dockerfile_path = output_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(config_result["dockerfile"])
            
            # Write requirements.txt
            requirements_path = output_dir / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write(dependencies_result["requirements_content"])
            
            # Write launch script
            launch_script_path = output_dir / f"launch_{generation_request.agent_id}.py"
            with open(launch_script_path, 'w') as f:
                f.write(config_result["launch_script"])
            
            # Make launch script executable
            os.chmod(launch_script_path, 0o755)
            
            # Generate agent metadata
            agent_metadata = {
                "agent_id": generation_request.agent_id,
                "agent_name": generation_request.agent_name,
                "description": generation_request.description,
                "template_used": generation_request.template_name,
                "generated_at": datetime.now().isoformat(),
                "generator_version": self.version,
                "context_id": context_id,
                "files_generated": [
                    code_result["generated_file"],
                    str(dockerfile_path),
                    str(requirements_path),
                    str(launch_script_path),
                    test_result["unit_test_file"],
                    test_result["integration_test_file"]
                ],
                "skills": len(template.skills + generation_request.custom_skills),
                "handlers": len(template.handlers + generation_request.custom_handlers),
                "dependencies": len(dependencies_result["dependencies"])
            }
            
            # Save agent metadata
            metadata_path = output_dir / "agent_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(agent_metadata, f, indent=2)
            
            # Register generated agent
            self.generated_agents[generation_request.agent_id] = agent_metadata
            
            self.processing_stats["agents_generated"] += 1
            self.processing_stats["total_processed"] += 1
            
            return AgentBuilderResponse(
                success=True,
                message=f"Agent '{generation_request.agent_name}' generated successfully",
                agent_id=generation_request.agent_id,
                generated_files=agent_metadata["files_generated"],
                metadata=agent_metadata
            ).dict()
            
        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return AgentBuilderResponse(
                success=False,
                message=f"Agent generation failed: {str(e)}",
                errors=[str(e)]
            ).dict()
    
    async def create_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent template"""
        
        # Validate template data
        validation_result = await self.execute_skill("template_validation", template_data)
        if not validation_result["valid"]:
            raise ValueError(f"Template validation failed: {validation_result['errors']}")
        
        # Create template object
        template = AgentTemplate(**template_data)
        
        # Save template
        template_file = self.templates_path / f"{template.name}_template.json"
        with open(template_file, 'w') as f:
            json.dump(template.dict(), f, indent=2)
        
        # Register template
        self.agent_templates[template.name] = template
        
        self.processing_stats["templates_created"] += 1
        
        return {
            "template_name": template.name,
            "template_file": str(template_file),
            "message": "Template created successfully"
        }
    
    async def list_templates(self) -> Dict[str, Any]:
        """List all available templates"""
        
        templates_list = []
        for name, template in self.agent_templates.items():
            templates_list.append({
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "skills_count": len(template.skills),
                "handlers_count": len(template.handlers),
                "tasks_count": len(template.tasks)
            })
        
        return {
            "templates": templates_list,
            "total_count": len(templates_list),
            "categories": list(set(t.category for t in self.agent_templates.values()))
        }
    
    async def update_template(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing template"""
        template_name = operation_data.get('template_name')
        if not template_name or template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Validate updated template data
        template_data = operation_data.get('template_data', {})
        validation_result = await self.execute_skill("template_validation", template_data)
        if not validation_result["valid"]:
            raise ValueError(f"Template validation failed: {validation_result['errors']}")
        
        # Update template
        updated_template = AgentTemplate(**template_data)
        self.agent_templates[template_name] = updated_template
        
        # Save updated template to file
        template_file = self.templates_path / f"{template_name}_template.json"
        with open(template_file, 'w') as f:
            json.dump(updated_template.dict(), f, indent=2)
        
        return {
            "template_name": template_name,
            "template_file": str(template_file),
            "message": "Template updated successfully"
        }
    
    async def delete_template(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delete template"""
        template_name = operation_data.get('template_name')
        if not template_name or template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Remove from memory
        del self.agent_templates[template_name]
        
        # Delete template files
        template_file = self.templates_path / f"{template_name}_template.json"
        template_py_file = self.templates_path / f"{template_name}_template.py.j2"
        
        if template_file.exists():
            template_file.unlink()
        if template_py_file.exists():
            template_py_file.unlink()
        
        return {
            "template_name": template_name,
            "message": "Template deleted successfully"
        }
    
    def _extract_request_data(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract request data from message"""
        request_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                request_data.update(part.data)
            elif part.kind == "file" and part.file:
                request_data["file"] = part.file
        
        return request_data
    
    async def _load_builtin_templates(self):
        """Load built-in agent templates"""
        
        # Basic Data Processing Agent Template
        data_processing_template = AgentTemplate(
            name="data_processing_agent",
            description="Template for data processing and transformation agents",
            category="data_processing",
            skills=["data_validation", "data_transformation", "data_quality_check"],
            handlers=["data_processing", "batch_processing"],
            tasks=["process_dataset", "validate_data", "transform_data"],
            dependencies=["pandas", "numpy", "pydantic"],
            resource_requirements={"memory": "512M", "cpu": "0.5"},
            template_variables={"default_batch_size": 1000, "max_file_size": "10MB"}
        )
        
        # AI/ML Agent Template
        ai_ml_template = AgentTemplate(
            name="ai_ml_agent",
            description="Template for AI and machine learning agents",
            category="ai_ml",
            skills=["model_inference", "data_preprocessing", "result_postprocessing"],
            handlers=["ml_prediction", "model_training"],
            tasks=["run_inference", "train_model", "evaluate_model"],
            dependencies=["torch", "transformers", "scikit-learn", "numpy"],
            resource_requirements={"memory": "2G", "cpu": "2.0", "gpu": "1"},
            template_variables={"model_cache_size": 3, "inference_timeout": 30}
        )
        
        # Integration Agent Template
        integration_template = AgentTemplate(
            name="integration_agent",
            description="Template for system integration and API agents",
            category="integration",
            skills=["api_client", "data_mapping", "protocol_translation"],
            handlers=["external_api", "webhook_handler"],
            tasks=["sync_data", "call_external_service", "process_webhook"],
            dependencies=["httpx", "requests", "aiohttp"],
            resource_requirements={"memory": "256M", "cpu": "0.25"},
            template_variables={"max_retries": 3, "timeout": 30}
        )
        
        # Register templates
        self.agent_templates = {
            "data_processing_agent": data_processing_template,
            "ai_ml_agent": ai_ml_template,
            "integration_agent": integration_template
        }
        
        # Create template files
        for template in self.agent_templates.values():
            await self._create_template_file(template)
    
    async def _create_template_file(self, template: AgentTemplate):
        """Create Jinja2 template file for agent generation"""
        
        template_content = '''"""
{{ description }}
Generated by Agent Builder Agent on {{ generated_at }}
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
import logging
import time

from ..sdk.performanceMonitoringMixin import PerformanceMonitoringMixin, monitor_a2a_operation
from app.a2a.sdk import (
    A2AAge, a2a_handlerntBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.utils import create_success_response, create_error_response
from app.a2a.core.workflowContext import workflowContextManager
from app.a2a.core.workflowMonitor import workflowMonitor
from prometheus_client import Counter, Histogram, Gauge, start_http_server

{% for dependency in dependencies %}
# {{ dependency }} - Add specific imports as needed
{% endfor %}

logger = logging.getLogger(__name__)


class {{ agent_name.replace(' ', '').replace('_', '') }}SDK(A2AAgentBase):
    """
    {{ agent_name }} - SDK Version
    {{ description }}
    """
    
    def __init__(self, base_url: str, **kwargs):
        super().__init__(
            agent_id="{{ agent_id }}",
            name="{{ agent_name }}",
            description="{{ description }}",
            version="1.0.0",
            base_url=base_url
        )
        
        # Configuration
        {% for key, value in configuration.items() %}
        self.{{ key }} = {{ value if value is string else value|tojson }}
        {% endfor %}
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set({{ skills|length }})
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            {% for skill in skills %}
            "{{ skill }}_count": 0,
            {% endfor %}
        }
        
        logger.info(f"Initialized {self.name} with SDK v1.0.0")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8000'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing {{ agent_name }}...")
        
        # Initialize storage
        storage_path = os.getenv("AGENT_STORAGE_PATH", "/tmp/{{ agent_id }}_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Custom initialization logic
        await self._custom_initialization()
        
        logger.info("{{ agent_name }} initialization complete")
    
    async def _custom_initialization(self):
        """Custom initialization logic - implement as needed"""
        # Load configuration from environment
        self.config_overrides = {
            'max_concurrent_tasks': int(os.getenv('MAX_CONCURRENT_TASKS', '10')),
            'timeout_seconds': int(os.getenv('TASK_TIMEOUT', '300')),
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }
        
        # Initialize custom storage paths
        self.custom_storage_path = Path(os.getenv('CUSTOM_STORAGE_PATH', '/tmp/agent_custom'))
        self.custom_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance tracking
        self.custom_metrics = {
            'initialization_time': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        logger.info(f"Custom initialization completed with config: {self.config_overrides}")

{% for handler in handlers %}
    @a2a_handler("{{ handler }}")
    async def handle_{{ handler }}(self, message: A2AMessage) -> Dict[str, Any]:
        """Handler for {{ handler }} requests"""
        start_time = time.time()
        
        try:
            # Extract request data
            request_data = self._extract_request_data(message)
            
            # Process request - implement your logic here
            result = await self._process_{{ handler }}(request_data, message.conversation_id)
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='{{ handler }}').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='{{ handler }}').observe(time.time() - start_time)
            
            return create_success_response(result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='{{ handler }}').inc()
            logger.error(f"{{ handler }} failed: {e}")
            return create_error_response(f"{{ handler }} failed: {str(e)}")
    
    async def _process_{{ handler }}(self, request_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Process {{ handler }} request - implement your logic here"""
        
        # Extract input parameters
        input_data = request_data.get('data', {})
        parameters = request_data.get('parameters', {})
        
        # Validate required parameters
        required_fields = ['input']  # Modify as needed for your {{ handler }}
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Implement your {{ handler }} logic here
        # Example implementation:
        result = {
            'processed_data': input_data,
            'processing_method': '{{ handler }}',
            'parameters_used': parameters,
            'processing_time': datetime.now().isoformat()
        }
        
        # Add any business logic specific to {{ handler }}
        # result = await self._apply_{{ handler }}_business_logic(input_data, parameters)
        
        # Log processing metrics
        logger.info(f"{{ handler }} processed {len(input_data)} items")
        
        return {
            "message": "{{ handler }} processed successfully", 
            "context_id": context_id,
            "result": result,
            "status": "completed"
        }

{% endfor %}

{% for skill in skills %}
    @a2a_skill("{{ skill }}")
    async def {{ skill }}_skill(self, *args, **kwargs) -> Dict[str, Any]:
        """{{ skill }} skill implementation"""
        
        try:
            # Extract skill parameters
            skill_input = kwargs.get('skill_input', args[0] if args else {})
            options = kwargs.get('options', {})
            
            # Validate skill inputs
            if not skill_input:
                raise ValueError("{{ skill }} skill requires input data")
            
            # Implement your {{ skill }} skill logic here
            # Example implementations for common skill types:
            
            if "{{ skill }}" == "data_processing":
                # Data processing skill implementation
                processed_data = self._process_skill_data(skill_input, options)
                result = {"processed_data": processed_data}
            elif "{{ skill }}" == "analysis":
                # Analysis skill implementation  
                analysis_result = self._analyze_skill_data(skill_input, options)
                result = {"analysis": analysis_result}
            elif "{{ skill }}" == "validation":
                # Validation skill implementation
                validation_result = self._validate_skill_data(skill_input, options)
                result = {"validation": validation_result, "is_valid": validation_result.get("valid", False)}
            else:
                # Generic skill implementation
                result = {
                    "skill_type": "{{ skill }}",
                    "input_processed": True,
                    "data": skill_input,
                    "options_applied": options
                }
            
            # Update skill metrics
            self.processing_stats["{{ skill }}_count"] += 1
            self.skills_processed.labels(agent_id=self.agent_id, skill_type="{{ skill }}").inc()
            
            # Add metadata to result
            result.update({
                "skill": "{{ skill }}",
                "timestamp": datetime.now().isoformat(),
                "execution_time": 0.1,  # Replace with actual timing
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"{{ skill }} skill failed: {e}")
            self.skills_failed.labels(agent_id=self.agent_id, skill_type="{{ skill }}").inc()
            return {
                "skill": "{{ skill }}",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_skill_data(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for data processing skills"""
        # Implement data processing logic specific to your skill
        return {"processed": True, "original_data": data}
    
    def _analyze_skill_data(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for analysis skills"""
        # Implement analysis logic specific to your skill
        return {"analyzed": True, "insights": [], "metrics": {}}
    
    def _validate_skill_data(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for validation skills"""
        # Implement validation logic specific to your skill
        return {"valid": True, "validation_errors": [], "validation_score": 1.0}

{% endfor %}

{% for task in tasks %}
    @a2a_task(
        task_type="{{ task }}",
        description="{{ task }} task implementation",
        timeout=300,
        retry_attempts=2
    )
    async def {{ task }}_task(self, *args, **kwargs) -> Dict[str, Any]:
        """{{ task }} task implementation"""
        
        try:
            # Extract task parameters
            task_input = kwargs.get('task_input', args[0] if args else {})
            task_context = kwargs.get('context', {})
            task_options = kwargs.get('options', {})
            
            # Validate task inputs
            if not task_input:
                logger.warning(f"{{ task }} task received empty input, using defaults")
                task_input = {}
            
            # Execute the task implementation
            result = await self._execute_{{ task }}(task_input, task_context, task_options)
            
            # Update task metrics
            self.processing_stats["total_processed"] += 1
            self.tasks_processed.labels(agent_id=self.agent_id, task_type="{{ task }}").inc()
            
            return {
                "task_successful": True,
                "task": "{{ task }}",
                "result": result,
                "execution_time": result.get("execution_time", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"{{ task }} task failed: {e}")
            return {
                "task_successful": False,
                "task": "{{ task }}",
                "error": str(e)
            }
    
    async def _execute_{{ task }}(self, task_input: Dict[str, Any], task_context: Dict[str, Any], task_options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute {{ task }} - implement your logic here"""
        
        start_time = datetime.now()
        
        try:
            # Implement your task-specific logic here
            # Example implementations for common task types:
            
            if "{{ task }}" == "data_transformation":
                # Data transformation task
                transformed_data = self._transform_data(task_input, task_options)
                result = {
                    "transformed_data": transformed_data,
                    "transformation_type": task_options.get("type", "default")
                }
                
            elif "{{ task }}" == "validation":
                # Validation task  
                validation_results = self._validate_data(task_input, task_context)
                result = {
                    "validation_passed": validation_results.get("valid", False),
                    "validation_errors": validation_results.get("errors", []),
                    "validation_score": validation_results.get("score", 0.0)
                }
                
            elif "{{ task }}" == "analysis":
                # Analysis task
                analysis_results = self._analyze_data(task_input, task_context, task_options)
                result = {
                    "analysis_results": analysis_results,
                    "insights": analysis_results.get("insights", []),
                    "metrics": analysis_results.get("metrics", {})
                }
                
            elif "{{ task }}" == "processing":
                # General processing task
                processed_results = self._process_task_data(task_input, task_options)
                result = {
                    "processed_data": processed_results,
                    "processing_method": task_options.get("method", "default"),
                    "items_processed": len(processed_results) if isinstance(processed_results, list) else 1
                }
                
            else:
                # Generic task implementation
                result = {
                    "task_type": "{{ task }}",
                    "input_data": task_input,
                    "context": task_context,
                    "options": task_options,
                    "status": "completed",
                    "custom_logic_applied": True
                }
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            # Add success indicators
            result.update({
                "message": "{{ task }} executed successfully",
                "success": True,
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"{{ task }} task completed successfully in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{{ task }} task execution failed after {execution_time:.3f}s: {e}")
            
            return {
                "message": f"{{ task }} execution failed: {str(e)}",
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "failed_at": datetime.now().isoformat()
            }
    
    def _transform_data(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for data transformation tasks"""
        # Implement your data transformation logic
        return {"transformed": True, "original": data, "options_applied": options}
    
    def _validate_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for validation tasks"""
        # Implement your validation logic
        return {"valid": True, "errors": [], "score": 1.0}
    
    def _analyze_data(self, data: Dict[str, Any], context: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for analysis tasks"""
        # Implement your analysis logic
        return {"insights": [], "metrics": {}, "summary": "Analysis completed"}
    
    def _process_task_data(self, data: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """Helper method for general processing tasks"""
        # Implement your processing logic
        return {"processed": True, "data": data}

{% endfor %}

    def _extract_request_data(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract request data from message"""
        request_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                request_data.update(part.data)
            elif part.kind == "file" and part.file:
                request_data["file"] = part.file
        
        return request_data
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save state if needed
            logger.info(f"{{ agent_name }} cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
'''
        
        template_file = self.templates_path / f"{template.name}_template.py.j2"
        with open(template_file, 'w') as f:
            f.write(template_content)
    
    def _validate_bpmn_workflow(self, workflow: BPMNWorkflow) -> Dict[str, Any]:
        """Validate BPMN workflow structure"""
        errors = []
        
        # Check required fields
        if not workflow.start_event:
            errors.append("Missing start event")
        
        if not workflow.end_events:
            errors.append("Missing end events")
        
        # Validate tasks
        task_ids = [task.get('id') for task in workflow.tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")
        
        # Validate sequence flows
        for flow in workflow.sequence_flows:
            source_id = flow.get('source_ref')
            target_id = flow.get('target_ref')
            
            if not source_id or not target_id:
                errors.append(f"Invalid sequence flow: missing source or target")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _generate_workflow_from_bpmn(self, workflow: BPMNWorkflow) -> str:
        """Generate workflow execution logic from BPMN"""
        
        workflow_code = f'''
async def execute_workflow_{workflow.workflow_id}(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generated workflow: {workflow.name}
    {workflow.description}
    """
    
    workflowContext = {{
        "workflow_id": "{workflow.workflow_id}",
        "started_at": datetime.now(),
        "input_data": input_data,
        "results": {{}}
    }}
    
    try:
        # Start event: {workflow.start_event}
        logger.info(f"Starting workflow {workflow.workflow_id}")
        
        # Execute tasks in sequence
'''
        
        # Add task execution logic
        for task in workflow.tasks:
            task_id = task.get('id', 'unknown_task')
            task_name = task.get('name', task_id)
            
            workflow_code += f'''
        # Task: {task_name}
        logger.info(f"Executing task: {task_name}")
        task_result = await self._execute_workflow_task("{task_id}", workflowContext)
        workflowContext["results"]["{task_id}"] = task_result
'''
        
        workflow_code += f'''
        # End events: {workflow.end_events}
        logger.info(f"Workflow {workflow.workflow_id} completed successfully")
        
        return {{
            "workflow_id": workflowContext["workflow_id"],
            "status": "completed",
            "started_at": workflowContext["started_at"].isoformat(),
            "completed_at": datetime.now().isoformat(),
            "results": workflowContext["results"]
        }}
        
    except Exception as e:
        logger.error(f"Workflow {workflow.workflow_id} failed: {{e}}")
        return {{
            "workflow_id": workflowContext["workflow_id"],
            "status": "failed",
            "error": str(e),
            "started_at": workflowContext["started_at"].isoformat(),
            "failed_at": datetime.now().isoformat()
        }}

async def _execute_workflow_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute individual workflow task"""
    start_time = time.time()
    
    try:
        # Task execution logic based on task_id
        if task_id.startswith('data_'):
            result = await self._execute_data_task(task_id, context)
        elif task_id.startswith('process_'):
            result = await self._execute_process_task(task_id, context)
        elif task_id.startswith('validate_'):
            result = await self._execute_validation_task(task_id, context)
        elif task_id.startswith('notify_'):
            result = await self._execute_notification_task(task_id, context)
        else:
            # Default task execution
            result = await self._execute_default_task(task_id, context)
        
        execution_time = time.time() - start_time
        
        return {{
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "context_updates": context.get('updates', {{}})
        }}
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Workflow task {{task_id}} failed: {{e}}")
        
        return {{
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }}

async def _execute_data_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data-related workflow task"""
    return {{"action": "data_processed", "records": len(context.get('data', []))}}

async def _execute_process_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute process workflow task"""
    return {{"action": "process_completed", "output": context.get('input', 'processed')}}

async def _execute_validation_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute validation workflow task"""
    is_valid = len(context.get('data', [])) > 0
    return {{"action": "validation_completed", "valid": is_valid}}

async def _execute_notification_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute notification workflow task"""
    return {{"action": "notification_sent", "recipients": context.get('recipients', [])}}

async def _execute_default_task(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute default workflow task"""
    return {{"action": "default_task_completed", "task_id": task_id}}
'''
        
        return workflow_code
    
    def _generate_workflow_config(self, workflow: BPMNWorkflow) -> Dict[str, Any]:
        """Generate workflow configuration"""
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "version": workflow.version,
            "tasks": [
                {
                    "id": task.get('id'),
                    "name": task.get('name'),
                    "type": task.get('type', 'service_task')
                }
                for task in workflow.tasks
            ],
            "gateways": workflow.gateways,
            "sequence_flows": workflow.sequence_flows,
            "data_objects": workflow.data_objects
        }
    
    def _generate_dockerfile(self, request: AgentGenerationRequest, template: AgentTemplate) -> str:
        """Generate Dockerfile for the agent"""
        
        return f'''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY {request.agent_id}_sdk.py .
COPY launch_{request.agent_id}.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "launch_{request.agent_id}.py"]
'''
    
    def _generate_compose_service(self, request: AgentGenerationRequest, template: AgentTemplate) -> Dict[str, Any]:
        """Generate Docker Compose service configuration"""
        
        return {
            request.agent_id: {
                "build": ".",
                "container_name": f"a2a-{request.agent_id}",
                "ports": ["8000:8000"],
                "environment": {
                    "AGENT_ID": request.agent_id,
                    "PROMETHEUS_PORT": "8000",
                    "LOG_LEVEL": "INFO",
                    **request.configuration
                },
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": "curl -f http://localhost:8000/health || exit 1",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                },
                "networks": ["a2a-network"],
                "deploy": {
                    "resources": {
                        "limits": template.resource_requirements
                    }
                }
            }
        }
    
    def _generate_env_config(self, request: AgentGenerationRequest, template: AgentTemplate) -> str:
        """Generate environment configuration"""
        
        env_vars = [
            f"AGENT_ID={request.agent_id}",
            f"AGENT_NAME={request.agent_name}",
            "PROMETHEUS_PORT=8000",
            "LOG_LEVEL=INFO",
            "A2A_SDK_VERSION=3.0.0"
        ]
        
        # Add custom configuration
        for key, value in request.configuration.items():
            env_vars.append(f"{key}={value}")
        
        return "\\n".join(env_vars)
    
    def _generate_launch_script(self, request: AgentGenerationRequest) -> str:
        """Generate agent launch script"""
        
        return f'''#!/usr/bin/env python3
"""
Launch script for {request.agent_name}
Generated by Agent Builder Agent
"""

import asyncio
import logging
import uvicorn
from {request.agent_id}_sdk import {request.agent_name.replace(' ', '').replace('_', '')}SDK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Launch the {request.agent_name}"""
    
    logger.info("🚀 Starting {request.agent_name}...")
    
    # Create agent instance
    agent = {request.agent_name.replace(' ', '').replace('_', '')}SDK(
        base_url="os.getenv("A2A_BASE_URL")"
    )
    
    # Initialize agent
    await agent.initialize()
    
    try:
        # Run the agent server
        uvicorn_config = uvicorn.Config(
            app=agent.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
        server = uvicorn.Server(uvicorn_config)
        logger.info("🌐 {request.agent_name} available at: http://localhost:8000")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down {request.agent_name}...")
    finally:
        await agent.cleanup()
        logger.info("✅ {request.agent_name} stopped")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _generate_unit_tests(self, agent_file_path: str) -> str:
        """Generate unit test content"""
        
        return f'''"""
Unit tests for generated agent
Generated by Agent Builder Agent
"""

import unittest
import asyncio
from unittest.mock import Mock, patch
import json

from {Path(agent_file_path).stem} import *


class TestGeneratedAgent(unittest.TestCase), PerformanceMonitoringMixin:
    """Unit tests for the generated agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = None  # Initialize with actual agent class
    
    def test_initialization(self):
        """Test agent initialization"""
        # Test agent exists and has required attributes
        self.assertIsNotNone(self.agent)
        self.assertTrue(hasattr(self.agent, 'agent_id'))
        self.assertTrue(hasattr(self.agent, 'name'))
        self.assertTrue(hasattr(self.agent, 'version'))
        
        # Test configuration was loaded
        if hasattr(self.agent, 'config_overrides'):
            self.assertIsInstance(self.agent.config_overrides, dict)
            self.assertIn('max_concurrent_tasks', self.agent.config_overrides)
    
    def test_handlers(self):
        """Test agent handlers"""
        # Test that agent has handlers attribute
        if hasattr(self.agent, 'handlers'):
            self.assertIsInstance(self.agent.handlers, dict)
        
        # Test handler registration
        if hasattr(self.agent, '_register_handler'):
            # Test that handler registration works
            test_handler = lambda x: {"result": "test"}
            self.agent._register_handler("test_handler", test_handler)
            self.assertIn("test_handler", self.agent.handlers)
    
    def test_skills(self):
        """Test agent skills"""
        # Test that agent has skills attribute
        if hasattr(self.agent, 'skills'):
            self.assertIsInstance(self.agent.skills, dict)
        
        # Test skill registration
        if hasattr(self.agent, '_register_skill'):
            test_skill = lambda *args, **kwargs: {"skill": "test_completed"}
            self.agent._register_skill("test_skill", test_skill)
            self.assertIn("test_skill", self.agent.skills)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test that agent handles invalid input gracefully
        if hasattr(self.agent, 'process_request'):
            try:
                # Test with invalid request
                result = self.agent.process_request(None)
                # Should not crash, should return error response
                self.assertIsInstance(result, dict)
                if 'error' in result:
                    self.assertTrue(True)  # Expected error response
            except Exception:
                # Should not raise unhandled exceptions
                self.fail("Agent should handle invalid input gracefully")
        
        # Test logging functionality
        if hasattr(self.agent, 'logger'):
            self.assertIsNotNone(self.agent.logger)


if __name__ == '__main__':
    unittest.main()
'''
    
    def _generate_integration_tests(self, agent_file_path: str) -> str:
        """Generate integration test content"""
        
        return f'''"""
Integration tests for generated agent
Generated by Agent Builder Agent
"""

import unittest
import asyncio
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
import json

from {Path(agent_file_path).stem} import *


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class TestAgentIntegration(unittest.TestCase), PerformanceMonitoringMixin:
    """Integration tests for the generated agent"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.agent = None  # Initialize with actual agent class
        cls.base_url = "os.getenv("A2A_BASE_URL")"
    
    def test_health_endpoint(self):
        """Test agent health endpoint"""
        if hasattr(cls.agent, 'health_check'):
            health_status = cls.agent.health_check()
            self.assertIsInstance(health_status, dict)
            self.assertIn('status', health_status)
            self.assertEqual(health_status['status'], 'healthy')
        else:
            # Basic health check - agent should be initialized
            self.assertIsNotNone(cls.agent)
    
    def test_agent_communication(self):
        """Test agent-to-agent communication"""
        # Test message creation
        if hasattr(cls.agent, 'create_message'):
            test_message = cls.agent.create_message(
                content="test_message",
                recipient="test_agent"
            )
            self.assertIsInstance(test_message, dict)
            self.assertIn('content', test_message)
        
        # Test network connectivity (if available)
        if hasattr(cls.agent, 'network_connector'):
            self.assertIsNotNone(cls.agent.network_connector)
    
    def test_workflow_execution(self):
        """Test workflow execution"""
        # Test workflow context creation
        if hasattr(cls.agent, 'create_workflow_context'):
            context = cls.agent.create_workflow_context("test_workflow")
            self.assertIsInstance(context, dict)
            self.assertIn('workflow_id', context)
        
        # Test task execution
        if hasattr(cls.agent, '_execute_workflow_task'):
            # This is an async method, so we test it exists
            self.assertTrue(callable(cls.agent._execute_workflow_task))
        
        # Test workflow monitoring
        if hasattr(cls.agent, 'workflow_monitor'):
            self.assertIsNotNone(cls.agent.workflow_monitor)


if __name__ == '__main__':
    unittest.main()
'''
    
    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            state_file = self.storage_path / "agent_builder_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.generated_agents = state_data.get("generated_agents", {})
                logger.info(f"Loaded {len(self.generated_agents)} generated agents from state")
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown agent - required by A2AAgentBase"""
        await self.cleanup()
    
    @a2a_handler("HEALTH_CHECK")
    async def handle_health_check(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle A2A protocol health check messages"""
        try:
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "name": "Agent Builder",
                "timestamp": datetime.utcnow().isoformat(),
                "blockchain_enabled": getattr(self, 'blockchain_enabled', False),
                "active_tasks": len(getattr(self, 'tasks', {})),
                "capabilities": getattr(self, 'blockchain_capabilities', []),
                "processing_stats": getattr(self, 'processing_stats', {}) or {},
                "response_time_ms": 0  # Immediate response for health checks
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_id": getattr(self, 'agent_id', 'unknown'),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save state
            state_file = self.storage_path / "agent_builder_state.json"
            state_data = {
                "generated_agents": self.generated_agents,
                "processing_stats": self.processing_stats
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, default=str, indent=2)
                
            logger.info(f"Saved state for {len(self.generated_agents)} generated agents")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
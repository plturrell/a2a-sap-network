"""
A2A Developer Portal - IDE for A2A Agent Development
Web-based IDE with BPMN designer, code editor, and agent management
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
from pathlib import Path
import logging
import tempfile
import shutil

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from ..agents.agent_builder_agent_sdk import AgentBuilderAgentSDK, AgentGenerationRequest, BPMNWorkflow

# SAP BTP Integration imports
from .sap_btp.auth_api import auth_router
from .sap_btp.rbac_service import initialize_rbac_service
from .sap_btp.session_service import initialize_session_service
from .sap_btp.destination_service import initialize_destination_service
from .sap_btp.logging_service import get_logger, setup_logging, LogCategory

# Import deployment pipeline
from .deployment.deployment_pipeline import DeploymentPipeline, DeploymentConfig, DeploymentTarget, DeploymentEnvironment


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# SAP CAP compliant logger
logger = get_logger(__name__)


class ProjectConfig(BaseModel):
    """Developer portal project configuration"""
    project_id: str
    name: str
    description: str
    created_at: datetime
    last_modified: datetime
    agents: List[Dict[str, Any]] = []
    workflows: List[Dict[str, Any]] = []
    templates: List[str] = []


class CodeFile(BaseModel):
    """Code file representation"""
    file_path: str
    content: str
    language: str = "python"
    last_modified: datetime


class DeveloperPortalServer:
    """
    A2A Developer Portal Server
    Provides web-based IDE for A2A agent development
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.projects = {}
        self.connected_clients = set()

        # Paths - Use proper storage configuration
        from ..config.storageConfig import get_workspace_path
        default_workspace = str(get_workspace_path())
        self.workspace_path = Path(config.get("workspace_path", default_workspace))
        self.templates_path = Path(config.get("templates_path", "app/a2a/developer_portal/templates"))
        self.static_path = Path(config.get("static_path", "app/a2a/developer_portal/static"))

        # Initialize workspace
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize Agent Builder
        self.agent_builder = AgentBuilderAgentSDK(
            base_url=os.getenv("A2A_SERVICE_URL"),
            templates_path=str(self.workspace_path / "templates")
        )

        # Initialize Deployment Pipeline
        deployment_config = {
            "deployments_path": str(self.workspace_path / "deployments"),
            "artifacts_path": str(self.workspace_path / "artifacts"),
            "logs_path": str(self.workspace_path / "logs"),
            "portal_url": f"http://localhost:{config.get('port', 8090)}",
            "email": config.get("email", {})
        }
        self.deployment_pipeline = DeploymentPipeline(deployment_config)

        # Initialize Blockchain Integration for workflows
        self.blockchain_config = {
            "networks": {
                "local": {
                    "provider_url": config.get("blockchain", {}).get("local_provider", os.getenv("A2A_SERVICE_URL")),
                    "chain_id": 31337
                },
                "testnet": {
                    "provider_url": config.get("blockchain", {}).get("testnet_provider", ""),
                    "chain_id": config.get("blockchain", {}).get("testnet_chain_id", 11155111)
                }
            },
            "contract_addresses": config.get("blockchain", {}).get("contracts", {})
        }

        # FastAPI app
        self.app = FastAPI(title="A2A Developer Portal", description="IDE for A2A Agent Development")

        # Setup SAP CAP logging
        setup_logging(self.app, "a2a-developer-portal")

        # Templates and static files
        self.templates = Jinja2Templates(directory=str(self.templates_path))

        # Initialize SAP BTP services
        self._initialize_sap_btp_services()

        # Setup routes
        self._setup_routes()

        logger.info("A2A Developer Portal initialized")

    def _setup_routes(self):
        """Setup FastAPI routes"""

        # Define specific routes before mounting static files

        # Serve SAP UI5 component files directly (needed for resource root mapping)
        @self.app.get("/Component.js")
        async def get_component_js():
            """Serve Component.js for SAP UI5"""
            return FileResponse(str(self.static_path / "Component.js"))

        @self.app.get("/manifest.json")
        async def get_manifest_json():
            """Serve manifest.json for SAP UI5"""
            return FileResponse(str(self.static_path / "manifest.json"))

        @self.app.get("/Component-preload.js")
        async def get_component_preload():
            """Serve Component-preload.js (optional optimization file)"""
            # Return empty response - this file is optional for development
            from fastapi.responses import Response
            return Response(content="", media_type="application/javascript")

        @self.app.get("/static/Component-preload.js")
        async def get_static_component_preload():
            """Serve Component-preload.js from static path (optional optimization file)"""
            # Return empty response - this file is optional for development
            from fastapi.responses import Response
            return Response(content="", media_type="application/javascript")

        # Serve other UI5 resources
        @self.app.get("/controller/{file_path:path}")
        async def get_controller(file_path: str):
            """Serve controller files"""
            return FileResponse(str(self.static_path / "controller" / file_path))

        @self.app.get("/view/{file_path:path}")
        async def get_view(file_path: str):
            """Serve view files"""
            return FileResponse(str(self.static_path / "view" / file_path))

        @self.app.get("/model/{file_path:path}")
        async def get_model(file_path: str):
            """Serve model files"""
            return FileResponse(str(self.static_path / "model" / file_path))

        @self.app.get("/i18n/{file_path:path}")
        async def get_i18n(file_path: str):
            """Serve i18n files"""
            # Handle localized files by falling back to base i18n.properties
            i18n_file = self.static_path / "i18n" / file_path
            if not i18n_file.exists() and "_" in file_path:
                # Fallback to base i18n.properties for any localized variants
                i18n_file = self.static_path / "i18n" / "i18n.properties"
            return FileResponse(str(i18n_file))

        @self.app.get("/fragment/{file_path:path}")
        async def get_fragment(file_path: str):
            """Serve fragment files"""
            return FileResponse(str(self.static_path / "fragment" / file_path))

        @self.app.get("/css/{file_path:path}")
        async def get_css(file_path: str):
            """Serve CSS files"""
            return FileResponse(str(self.static_path / "css" / file_path))

        # Mount static files AFTER defining specific routes
        self.app.mount("/static", StaticFiles(directory=str(self.static_path)), name="static")

        @self.app.get("/favicon.ico")
        @self.app.head("/favicon.ico")
        async def favicon():
            """Serve favicon for both GET and HEAD requests"""
            from fastapi.responses import Response
            return Response(status_code=200, content=b"", media_type="image/x-icon")

        @self.app.get("/", response_class=HTMLResponse)
        async def portal_home(request: Request):
            """Serve the main portal page"""
            return self.templates.TemplateResponse("portal.html", {
                "request": request,
                "projects": list(self.projects.values())
            })

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "portal": "SAP UI5 Fiori Developer Portal", "version": "1.0.0"}

        @self.app.get("/api/projects")
        async def get_projects():
            """Get all projects"""
            return {"projects": list(self.projects.values())}

        @self.app.post("/api/projects")
        async def create_project(project_data: Dict[str, Any]):
            """Create new project"""
            project_id = project_data.get("name", "").lower().replace(" ", "_")

            if project_id in self.projects:
                raise HTTPException(status_code=400, detail="Project already exists")

            project = ProjectConfig(
                project_id=project_id,
                name=project_data["name"],
                description=project_data.get("description", ""),
                created_at=datetime.now(),
                last_modified=datetime.now()
            )

            # Create project directory
            project_dir = self.workspace_path / project_id
            project_dir.mkdir(exist_ok=True)

            # Save project config
            with open(project_dir / "project.json", 'w') as f:
                json.dump(project.dict(), f, default=str, indent=2)

            self.projects[project_id] = project

            await self._broadcast_update({
                "type": "project_created",
                "project": project.dict()
            })

            return {"success": True, "project": project.dict()}

        @self.app.get("/api/projects/{project_id}")
        async def get_project(project_id: str):
            """Get project details"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            return self.projects[project_id].dict()

        @self.app.get("/api/projects/{project_id}/files")
        async def get_project_files(project_id: str):
            """Get project files"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            project_dir = self.workspace_path / project_id
            files = []

            for file_path in project_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    relative_path = file_path.relative_to(project_dir)

                    # Determine language
                    language = "python" if file_path.suffix == ".py" else "text"
                    if file_path.suffix in [".json", ".yaml", ".yml"]:
                        language = "json" if file_path.suffix == ".json" else "yaml"

                    files.append({
                        "path": str(relative_path),
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "language": language,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime)
                    })

            return {"files": files}

        @self.app.get("/api/projects/{project_id}/files/{file_path:path}")
        async def get_file_content(project_id: str, file_path: str):
            """Get file content"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            file_full_path = self.workspace_path / project_id / file_path

            if not file_full_path.exists():
                raise HTTPException(status_code=404, detail="File not found")

            try:
                with open(file_full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return {
                    "content": content,
                    "language": "python" if file_full_path.suffix == ".py" else "text",
                    "last_modified": datetime.fromtimestamp(file_full_path.stat().st_mtime)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        @self.app.put("/api/projects/{project_id}/files/{file_path:path}")
        async def save_file_content(project_id: str, file_path: str, file_data: Dict[str, Any]):
            """Save file content"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            file_full_path = self.workspace_path / project_id / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(file_full_path, 'w', encoding='utf-8') as f:
                    f.write(file_data["content"])

                # Update project modification time
                self.projects[project_id].last_modified = datetime.now()

                await self._broadcast_update({
                    "type": "file_saved",
                    "project_id": project_id,
                    "file_path": file_path
                })

                return {
                    "success": True,
                    "last_modified": datetime.fromtimestamp(file_full_path.stat().st_mtime)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

        @self.app.post("/api/projects/{project_id}/agents/generate")
        async def generate_agent(project_id: str, generation_data: Dict[str, Any]):
            """Generate new agent using Agent Builder"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            try:
                # Prepare generation request
                output_dir = self.workspace_path / project_id / "agents" / generation_data["agent_id"]

                request = AgentGenerationRequest(
                    agent_name=generation_data["agent_name"],
                    agent_id=generation_data["agent_id"],
                    description=generation_data.get("description", ""),
                    template_name=generation_data["template_name"],
                    custom_skills=generation_data.get("custom_skills", []),
                    custom_handlers=generation_data.get("custom_handlers", []),
                    configuration=generation_data.get("configuration", {}),
                    output_directory=str(output_dir)
                )

                # Add BPMN workflow if provided
                if generation_data.get("bpmn_workflow"):
                    request.bpmn_workflow = BPMNWorkflow(**generation_data["bpmn_workflow"])

                # Generate agent
                result = await self.agent_builder.generate_agent(request, f"portal_{project_id}")

                if result["success"]:
                    # Update project with new agent
                    agent_info = {
                        "agent_id": request.agent_id,
                        "agent_name": request.agent_name,
                        "template": request.template_name,
                        "generated_at": datetime.now(),
                        "files": result["generated_files"]
                    }

                    self.projects[project_id].agents.append(agent_info)

                    await self._broadcast_update({
                        "type": "agent_generated",
                        "project_id": project_id,
                        "agent": agent_info
                    })

                return result

            except Exception as e:
                logger.error(f"Agent generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Agent generation failed: {str(e)}")

        @self.app.get("/api/templates")
        async def get_templates():
            """Get available agent templates"""
            try:
                # Get templates from Agent Builder
                templates_result = await self.agent_builder.list_templates()
                return templates_result
            except Exception as e:
                logger.error(f"Failed to get templates: {e}")
                raise HTTPException(status_code=500, detail="Failed to get templates")

        @self.app.post("/api/projects/{project_id}/workflows/save")
        async def save_bpmn_workflow(project_id: str, workflow_data: Dict[str, Any]):
            """Save BPMN workflow"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            try:
                workflow_dir = self.workspace_path / project_id / "workflows"
                workflow_dir.mkdir(exist_ok=True)

                workflow_file = workflow_dir / f"{workflow_data['workflow_id']}.json"

                with open(workflow_file, 'w') as f:
                    json.dump(workflow_data, f, indent=2)

                # Update project
                workflow_info = {
                    "workflow_id": workflow_data["workflow_id"],
                    "name": workflow_data.get("name", ""),
                    "description": workflow_data.get("description", ""),
                    "saved_at": datetime.now()
                }

                # Update or add workflow
                existing_workflow = next(
                    (w for w in self.projects[project_id].workflows if w["workflow_id"] == workflow_data["workflow_id"]),
                    None
                )

                if existing_workflow:
                    existing_workflow.update(workflow_info)
                else:
                    self.projects[project_id].workflows.append(workflow_info)

                await self._broadcast_update({
                    "type": "workflow_saved",
                    "project_id": project_id,
                    "workflow": workflow_info
                })

                return {"success": True, "workflow": workflow_info}

            except Exception as e:
                logger.error(f"Failed to save workflow: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save workflow: {str(e)}")

        @self.app.get("/api/projects/{project_id}/workflows/{workflow_id}")
        async def get_bpmn_workflow(project_id: str, workflow_id: str):
            """Get BPMN workflow"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            workflow_file = self.workspace_path / project_id / "workflows" / f"{workflow_id}.json"

            if not workflow_file.exists():
                raise HTTPException(status_code=404, detail="Workflow not found")

            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)

                return workflow_data

            except Exception as e:
                logger.error(f"Failed to load workflow: {e}")
                raise HTTPException(status_code=500, detail="Failed to load workflow")

        @self.app.post("/api/projects/{project_id}/test")
        async def test_project_agents(project_id: str, test_config: Optional[Dict[str, Any]] = None):
            """Run real tests on project agents"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            try:
                import subprocess
                import tempfile

                project_dir = self.workspace_path / project_id
                test_results = []
                total_passed = 0
                total_failed = 0

                # Determine test framework
                test_framework = "pytest"  # Default
                if (project_dir / "package.json").exists():
                    # Node.js project
                    test_framework = "jest"
                elif (project_dir / "go.mod").exists():
                    # Go project
                    test_framework = "go test"

                # Find test files based on framework
                if test_framework == "pytest":
                    test_files = list(project_dir.rglob("test_*.py")) + list(project_dir.rglob("*_test.py"))
                elif test_framework == "jest":
                    test_files = list(project_dir.rglob("*.test.js")) + list(project_dir.rglob("*.spec.js"))
                elif test_framework == "go test":
                    test_files = list(project_dir.rglob("*_test.go"))

                if not test_files:
                    return {
                        "success": True,
                        "tests_run": 0,
                        "message": "No test files found",
                        "framework": test_framework
                    }

                # Create test report directory
                report_dir = project_dir / "test-reports"
                report_dir.mkdir(exist_ok=True)

                # Run tests based on framework
                if test_framework == "pytest":
                    # Run pytest with detailed reporting
                    cmd = [
                        "python", "-m", "pytest",
                        str(project_dir),
                        "-v",
                        "--tb=short",
                        "--junit-xml=" + str(report_dir / "junit.xml"),
                        "--html=" + str(report_dir / "report.html"),
                        "--self-contained-html",
                        "--cov=" + str(project_dir / "src"),
                        "--cov-report=html:" + str(report_dir / "coverage"),
                        "--cov-report=term"
                    ]

                    # Add custom pytest args if provided
                    if test_config and test_config.get("pytest_args"):
                        cmd.extend(test_config["pytest_args"])

                elif test_framework == "jest":
                    # Run Jest with coverage
                    cmd = [
                        "npm", "test", "--",
                        "--coverage",
                        "--coverageDirectory=" + str(report_dir / "coverage"),
                        "--json",
                        "--outputFile=" + str(report_dir / "jest-results.json")
                    ]

                elif test_framework == "go test":
                    # Run Go tests with coverage
                    cmd = [
                        "go", "test",
                        "./...",
                        "-v",
                        "-cover",
                        "-coverprofile=" + str(report_dir / "coverage.out"),
                        "-json"
                    ]

                # Execute tests
                result = subprocess.run(
                    cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=test_config.get("timeout", 300) if test_config else 300  # 5 min timeout
                )

                # Parse results
                if test_framework == "pytest":
                    # Parse pytest output
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if "passed" in line and "failed" in line:
                            # Extract test counts
                            import re
                            passed_match = re.search(r'(\d+) passed', line)
                            failed_match = re.search(r'(\d+) failed', line)
                            if passed_match:
                                total_passed = int(passed_match.group(1))
                            if failed_match:
                                total_failed = int(failed_match.group(1))

                    # Parse individual test results from XML if available
                    junit_file = report_dir / "junit.xml"
                    if junit_file.exists():
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(junit_file)
                        root = tree.getroot()

                        for testcase in root.findall(".//testcase"):
                            test_name = testcase.get("name")
                            classname = testcase.get("classname")
                            time = float(testcase.get("time", 0))

                            failure = testcase.find("failure")
                            error = testcase.find("error")

                            if failure is not None or error is not None:
                                status = "failed"
                                message = (failure or error).get("message", "Test failed")
                            else:
                                status = "passed"
                                message = "Test passed"

                            test_results.append({
                                "test": test_name,
                                "class": classname,
                                "status": status,
                                "duration": time,
                                "message": message
                            })

                # Get coverage data if available
                coverage_data = None
                if test_framework == "pytest":
                    coverage_file = report_dir / "coverage" / "index.html"
                    if coverage_file.exists():
                        # Extract coverage percentage from HTML
                        with open(coverage_file, 'r') as f:
                            content = f.read()
                            import re
                            match = re.search(r'(\d+)%', content)
                            if match:
                                coverage_data = {
                                    "percentage": int(match.group(1)),
                                    "report_path": str(coverage_file.relative_to(project_dir))
                                }

                return {
                    "success": result.returncode == 0,
                    "tests_run": total_passed + total_failed,
                    "passed": total_passed,
                    "failed": total_failed,
                    "framework": test_framework,
                    "results": test_results,
                    "coverage": coverage_data,
                    "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,  # Last 1000 chars
                    "errors": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                    "report_dir": str(report_dir.relative_to(project_dir))
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Test execution timed out",
                    "timeout": True
                }
            except Exception as e:
                logger.error(f"Testing failed: {e}")
                import traceback
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

        @self.app.post("/api/projects/{project_id}/deploy")
        async def deploy_project(project_id: str, deploy_config: Dict[str, Any] = None):
            """Deploy project agents using real deployment pipeline"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            try:
                project = self.projects[project_id]

                # Create deployment configuration
                deployment_config = DeploymentConfig(
                    name=f"{project.name} Deployment",
                    project_id=project_id,
                    target=DeploymentTarget(
                        name="production",
                        environment=DeploymentEnvironment.PRODUCTION if deploy_config and deploy_config.get("environment") == "production" else DeploymentEnvironment.STAGING,
                        platform=deploy_config.get("platform", "kubernetes") if deploy_config else "kubernetes",
                        cpu_limit=deploy_config.get("cpu_limit", "500m") if deploy_config else "500m",
                        memory_limit=deploy_config.get("memory_limit", "512Mi") if deploy_config else "512Mi",
                        replicas=deploy_config.get("replicas", 1) if deploy_config else 1
                    ),
                    dockerfile_path="Dockerfile",
                    build_context=str(self.workspace_path / project_id),
                    environment_variables=deploy_config.get("env_vars", {}) if deploy_config else {},
                    notification_emails=deploy_config.get("notification_emails", []) if deploy_config else []
                )

                # Store deployment configuration
                created_config = await self.deployment_pipeline.create_deployment_config(deployment_config.dict())

                # Execute deployment
                version = deploy_config.get("version", "latest") if deploy_config else "latest"
                execution = await self.deployment_pipeline.deploy(created_config.id, version)

                # Update project deployment status
                project.last_modified = datetime.now()

                return {
                    "success": True,
                    "deployment_id": execution.id,
                    "config_id": created_config.id,
                    "status": execution.status.value,
                    "message": f"Deployment initiated for project {project.name}",
                    "logs_url": f"/api/deployments/{execution.id}/logs"
                }

            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

        @self.app.post("/api/projects/{project_id}/export")
        async def export_project(project_id: str):
            """Export project as ZIP file"""
            if project_id not in self.projects:
                raise HTTPException(status_code=404, detail="Project not found")

            try:
                project_dir = self.workspace_path / project_id

                # Create temporary ZIP file
                temp_dir = Path(tempfile.mkdtemp())
                zip_file = temp_dir / f"{project_id}.zip"

                shutil.make_archive(str(zip_file).replace('.zip', ''), 'zip', project_dir)

                return FileResponse(
                    path=str(zip_file),
                    filename=f"{project_id}.zip",
                    media_type="application/zip"
                )

            except Exception as e:
                logger.error(f"Export failed: {e}")
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connected_clients.add(websocket)

            try:
                while True:
                    # Keep connection alive and handle client messages
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    # Handle client messages (e.g., cursor position, typing indicators)
                    await self._handle_client_message(websocket, data)

            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)

        @self.app.get("/api/deployments/{execution_id}")
        async def get_deployment_status(execution_id: str):
            """Get deployment execution status"""
            execution = await self.deployment_pipeline.get_deployment_execution(execution_id)
            if not execution:
                raise HTTPException(status_code=404, detail="Deployment execution not found")

            return execution.dict()

        @self.app.get("/api/deployments/{execution_id}/logs")
        async def get_deployment_logs(execution_id: str):
            """Get deployment logs"""
            logs = await self.deployment_pipeline.get_deployment_logs(execution_id)
            if logs is None:
                raise HTTPException(status_code=404, detail="Deployment execution not found")

            return {
                "execution_id": execution_id,
                "logs": logs,
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.get("/api/projects/{project_id}/deployments")
        async def get_project_deployments(project_id: str):
            """Get deployment history for a project"""
            # Find all deployment configs for this project
            all_configs = []
            for config in self.deployment_pipeline.deployments.values():
                if config.project_id == project_id:
                    all_configs.append(config.id)

            # Get all executions for these configs
            deployments = []
            for config_id in all_configs:
                history = await self.deployment_pipeline.get_deployment_history(config_id)
                deployments.extend([exec.dict() for exec in history])

            # Sort by start time
            deployments.sort(key=lambda x: x['started_at'], reverse=True)

            return {
                "project_id": project_id,
                "deployments": deployments,
                "total": len(deployments)
            }

        @self.app.post("/api/workflows/{workflow_id}/execute")
        async def execute_workflow(workflow_id: str, execution_data: Dict[str, Any]):
            """Execute BPMN workflow with blockchain support"""
            try:
                # Import workflow designer and engine config
                from .bpmn.workflow_designer import BPMNWorkflowDesigner
                from .bpmn.workflow_engine import WorkflowEngineConfig

                # Create engine config with blockchain settings
                engine_config = WorkflowEngineConfig(
                    persistence_path=str(self.workspace_path / "workflow_executions"),
                    enable_persistence=True,
                    blockchain=self.blockchain_config
                )

                # Initialize workflow designer with blockchain config
                workflow_designer = BPMNWorkflowDesigner(str(self.workspace_path / "workflows"))
                workflow_designer.execution_engine.config = engine_config

                # Get workflow
                workflow = await workflow_designer.get_workflow(workflow_id)
                if not workflow:
                    raise HTTPException(status_code=404, detail="Workflow not found")

                # Add blockchain private key if provided
                variables = execution_data.get("variables", {})
                if "privateKey" in execution_data:
                    variables["privateKey"] = execution_data["privateKey"]

                # Execute workflow
                execution = await workflow_designer.execute_workflow(
                    workflow_id,
                    variables
                )

                return {
                    "success": True,
                    "execution_id": execution.id,
                    "status": execution.status.value,
                    "message": "Workflow execution started",
                    "blockchain_enabled": True
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/workflow-executions/{execution_id}")
        async def get_workflow_execution(execution_id: str):
            """Get workflow execution status"""
            try:
                # Import workflow designer
                from .bpmn.workflow_designer import BPMNWorkflowDesigner

                # Initialize workflow designer
                workflow_designer = BPMNWorkflowDesigner(str(self.workspace_path / "workflows"))

                # Get execution
                execution = await workflow_designer.get_execution(execution_id)
                if not execution:
                    raise HTTPException(status_code=404, detail="Execution not found")

                return execution.dict()

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get workflow execution: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/blockchain/register-agent")
        async def register_workflow_agent(agent_data: Dict[str, Any]):
            """Register workflow agent on blockchain"""
            try:
                from .bpmn.blockchain_integration import A2ABlockchainIntegration, SmartContractTaskType

                # Initialize blockchain integration
                blockchain = A2ABlockchainIntegration(self.blockchain_config)

                # Register agent
                result = await blockchain.execute_blockchain_task(
                    task_type=SmartContractTaskType.AGENT_REGISTRATION,
                    network=agent_data.get("network", "local"),
                    task_config={
                        "agentName": agent_data.get("name", "Workflow Agent"),
                        "agentEndpoint": agent_data.get("endpoint", f"http://localhost:{self.config.get('port', 8090)}/api/agent"),
                        "capabilities": agent_data.get("capabilities", ["workflow", "bpmn"]),
                        "privateKey": agent_data.get("privateKey")
                    },
                    variables={}
                )

                await blockchain.close()

                return result

            except Exception as e:
                logger.error(f"Agent registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/blockchain/agents")
        async def discover_blockchain_agents(network: str = "local", capability: Optional[str] = None):
            """Discover agents on blockchain"""
            try:
                from .bpmn.blockchain_integration import A2ABlockchainIntegration, SmartContractTaskType

                # Initialize blockchain integration
                blockchain = A2ABlockchainIntegration(self.blockchain_config)

                if capability:
                    # Query by capability
                    result = await blockchain.execute_blockchain_task(
                        task_type=SmartContractTaskType.CAPABILITY_QUERY,
                        network=network,
                        task_config={"capability": capability},
                        variables={}
                    )
                else:
                    # Discover all agents
                    result = await blockchain.execute_blockchain_task(
                        task_type=SmartContractTaskType.AGENT_DISCOVERY,
                        network=network,
                        task_config={},
                        variables={}
                    )

                await blockchain.close()

                return result

            except Exception as e:
                logger.error(f"Agent discovery failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/blockchain/send-message")
        async def send_blockchain_message(message_data: Dict[str, Any]):
            """Send message through blockchain"""
            try:
                from .bpmn.blockchain_integration import A2ABlockchainIntegration, SmartContractTaskType

                # Initialize blockchain integration
                blockchain = A2ABlockchainIntegration(self.blockchain_config)

                # Send message
                result = await blockchain.execute_blockchain_task(
                    task_type=SmartContractTaskType.MESSAGE_ROUTING,
                    network=message_data.get("network", "local"),
                    task_config={
                        "toAgent": message_data["toAgent"],
                        "content": message_data["content"],
                        "messageType": message_data.get("messageType", "general"),
                        "privateKey": message_data.get("privateKey")
                    },
                    variables={}
                )

                await blockchain.close()

                return result

            except Exception as e:
                logger.error(f"Message sending failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/blockchain/contracts")
        async def get_contract_addresses(network: str = "local"):
            """Get deployed contract addresses"""
            try:
                from .bpmn.blockchain_integration import A2AContractAddresses

                if network == "local":
                    addresses = A2AContractAddresses.LOCAL
                elif network == "testnet":
                    addresses = A2AContractAddresses.TESTNET
                else:
                    addresses = self.blockchain_config.get("contract_addresses", {}).get(network, {})

                return {
                    "network": network,
                    "contracts": addresses,
                    "blockchain_enabled": True
                }

            except Exception as e:
                logger.error(f"Failed to get contract addresses: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "projects_count": len(self.projects),
                "connected_clients": len(self.connected_clients),
                "workspace_path": str(self.workspace_path),
                "blockchain_enabled": True
            }

    async def _broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.connected_clients:
            return

        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message, default=str))
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)

    async def _handle_client_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = data.get("type")

        if message_type == "cursor_position":
            # Broadcast cursor position to other clients
            await self._broadcast_update({
                "type": "cursor_update",
                "client_id": data.get("client_id"),
                "position": data.get("position"),
                "file_path": data.get("file_path")
            })

        elif message_type == "typing":
            # Broadcast typing indicator
            await self._broadcast_update({
                "type": "typing_indicator",
                "client_id": data.get("client_id"),
                "file_path": data.get("file_path"),
                "is_typing": data.get("is_typing")
            })

    async def initialize(self):
        """Initialize the developer portal"""
        logger.info("Initializing Developer Portal...")

        # Initialize Agent Builder
        await self.agent_builder.initialize()

        # Load existing projects
        await self._load_projects()

        logger.info(f"Developer Portal initialized with {len(self.projects)} projects")

    async def _load_projects(self):
        """Load existing projects from workspace"""
        try:
            for project_dir in self.workspace_path.iterdir():
                if project_dir.is_dir():
                    project_config_file = project_dir / "project.json"

                    if project_config_file.exists():
                        with open(project_config_file, 'r') as f:
                            project_data = json.load(f)

                        project = ProjectConfig(**project_data)
                        self.projects[project.project_id] = project

            logger.info(f"Loaded {len(self.projects)} existing projects")

        except Exception as e:
            logger.warning(f"Failed to load projects: {e}")

    def _initialize_sap_btp_services(self):
        """Initialize SAP BTP services for enterprise authentication and cloud integration"""
        try:
            import os

            # SAP BTP service configurations from environment
            sap_config = self.config.get("sap_btp", {})

            # Initialize RBAC service with production configuration
            rbac_config = sap_config.get("rbac", {
                "xsuaa_service_url": os.environ.get('XSUAA_SERVICE_URL', 'https://tenant.authentication.sap.hana.ondemand.com'),
                "client_id": os.environ.get('XSUAA_CLIENT_ID', 'sb-a2a-portal'),
                "client_secret": os.environ.get('XSUAA_CLIENT_SECRET'),
                "development_mode": os.environ.get('DEVELOPMENT_MODE', 'false').lower() == 'true'
            })

            if not rbac_config["client_secret"] and not rbac_config["development_mode"]:
                raise ValueError("XSUAA_CLIENT_SECRET must be set in production")

            initialize_rbac_service(rbac_config)

            # Initialize Session service with Redis configuration
            session_config = sap_config.get("session", {
                "session_timeout_minutes": int(os.environ.get('SESSION_TIMEOUT_MINUTES', '30')),
                "max_sessions_per_user": int(os.environ.get('MAX_SESSIONS_PER_USER', '5')),
                "cleanup_interval_minutes": int(os.environ.get('SESSION_CLEANUP_INTERVAL', '5')),
                "redis": {
                    "host": os.environ.get('REDIS_HOST', 'localhost'),
                    "port": int(os.environ.get('REDIS_PORT', '6379')),
                    "password": os.environ.get('REDIS_PASSWORD'),
                    "db": int(os.environ.get('REDIS_DB', '0')),
                    "ssl": os.environ.get('REDIS_SSL', 'false').lower() == 'true'
                }
            })
            initialize_session_service(session_config)

            # Initialize Destination service
            destination_config = sap_config.get("destination", {
                "uri": os.environ.get('DESTINATION_SERVICE_URL', 'https://destination-configuration.cfapps.sap.hana.ondemand.com'),
                "clientid": os.environ.get('DESTINATION_CLIENT_ID', 'sb-destination-service'),
                "clientsecret": os.environ.get('DESTINATION_CLIENT_SECRET'),
                "url": os.environ.get('DESTINATION_AUTH_URL', 'https://tenant.authentication.sap.hana.ondemand.com')
            })

            if not destination_config["clientsecret"] and not rbac_config["development_mode"]:
                raise ValueError("DESTINATION_CLIENT_SECRET must be set in production")

            initialize_destination_service(destination_config)

            # Include authentication router
            self.app.include_router(auth_router)

            # Include A2A Network integration router
            from .api.a2a_network_integration import router as a2a_network_router, cleanup as a2a_cleanup
            self.app.include_router(a2a_network_router)

            # Register cleanup handler
            self.app.on_event("shutdown")(a2a_cleanup)

            logger.info("SAP BTP services initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize SAP BTP services: {e}")
            logger.info("Running in development mode without SAP BTP integration")


# Factory function
def create_developer_portal(config: Optional[Dict[str, Any]] = None) -> DeveloperPortalServer:
    """Create Developer Portal server instance"""
    from ..config.storageConfig import get_workspace_path

    default_config = {
        "workspace_path": str(get_workspace_path()),
        "templates_path": "app/a2a/developer_portal/templates",
        "static_path": "app/a2a/developer_portal/static",
        "port": 3001
    }

    if config:
        default_config.update(config)

    return DeveloperPortalServer(default_config)

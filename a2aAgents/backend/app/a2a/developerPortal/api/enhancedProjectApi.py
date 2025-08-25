"""
Enhanced Project API Endpoints for A2A Developer Portal
Provides comprehensive project management with real data connections
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models.project_models import (
    ProjectDataManager, EnhancedProject, ProjectStatus, ProjectType,
    DeploymentStatus, ProjectMetrics, ProjectDependency, DeploymentConfig
)
from ..sap_btp.auth_api import get_current_user, UserInfo

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v2", tags=["Enhanced Projects"])

# Global project data manager (will be initialized by the portal server)
project_manager: Optional[ProjectDataManager] = None


def get_project_manager() -> ProjectDataManager:
    """Dependency to get project manager"""
    if project_manager is None:
        raise HTTPException(status_code=500, detail="Project manager not initialized")
    return project_manager


class ProjectCreateRequest(BaseModel):
    """Request model for creating projects"""
    name: str
    description: str = ""
    project_type: ProjectType = ProjectType.AGENT
    tags: List[str] = []
    template_id: Optional[str] = None


class ProjectUpdateRequest(BaseModel):
    """Request model for updating projects"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    tags: Optional[List[str]] = None
    deployment_config: Optional[Dict[str, Any]] = None


class ProjectSearchRequest(BaseModel):
    """Request model for searching projects"""
    query: Optional[str] = None
    project_type: Optional[ProjectType] = None
    status: Optional[ProjectStatus] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


@router.get("/projects")
async def get_projects(
    project_type: Optional[ProjectType] = Query(None),
    status: Optional[ProjectStatus] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get all projects with filtering and pagination"""
    try:
        all_projects = await manager.get_all_projects()

        # Apply filters
        filtered_projects = all_projects
        if project_type:
            filtered_projects = [p for p in filtered_projects if p.project_type == project_type]
        if status:
            filtered_projects = [p for p in filtered_projects if p.status == status]

        # Apply pagination
        total = len(filtered_projects)
        projects = filtered_projects[offset:offset + limit]

        # Convert to dict format for UI5
        projects_data = []
        for project in projects:
            project_dict = project.dict()
            project_dict["metrics"] = await manager.get_project_metrics(project.id)
            projects_data.append(project_dict)

        return {
            "projects": projects_data,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }

    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects")
async def create_project(
    request: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    manager: ProjectDataManager = Depends(get_project_manager),
    current_user: UserInfo = Depends(get_current_user)
):
    """Create a new project"""
    try:
        project_data = request.dict()

        # Add default configuration
        project_data["deployment_config"] = DeploymentConfig().dict()
        project_data["created_by"] = current_user.email  # Get from authenticated user

        # Create project
        project = await manager.create_project(project_data)

        # Initialize project structure in background
        background_tasks.add_task(initialize_project_structure, project.id, request.template_id)

        return {
            "success": True,
            "project": project.dict(),
            "message": "Project created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get project details"""
    try:
        project = await manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get enhanced data
        project_dict = project.dict()
        project_dict["metrics"] = await manager.get_project_metrics(project_id)

        return project_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}")
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Update project"""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}

        project = await manager.update_project(project_id, updates)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "success": True,
            "project": project.dict(),
            "message": "Project updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Delete project"""
    try:
        success = await manager.delete_project(project_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "success": True,
            "message": "Project deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/metrics")
async def get_project_metrics(
    project_id: str,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get project performance metrics"""
    try:
        metrics = await manager.get_project_metrics(project_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Project not found")

        return metrics.dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/search")
async def search_projects(
    request: ProjectSearchRequest,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Search projects with advanced filtering"""
    try:
        all_projects = await manager.get_all_projects()

        # Apply search filters
        filtered_projects = all_projects

        if request.query:
            query_lower = request.query.lower()
            filtered_projects = [
                p for p in filtered_projects
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]

        if request.project_type:
            filtered_projects = [p for p in filtered_projects if p.project_type == request.project_type]

        if request.status:
            filtered_projects = [p for p in filtered_projects if p.status == request.status]

        if request.tags:
            filtered_projects = [
                p for p in filtered_projects
                if any(tag in p.tags for tag in request.tags)
            ]

        if request.created_after:
            filtered_projects = [p for p in filtered_projects if p.created_at >= request.created_after]

        if request.created_before:
            filtered_projects = [p for p in filtered_projects if p.created_at <= request.created_before]

        # Convert to dict format
        projects_data = [project.dict() for project in filtered_projects]

        return {
            "projects": projects_data,
            "total": len(projects_data),
            "query": request.dict()
        }

    except Exception as e:
        logger.error(f"Error searching projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/agents")
async def get_project_agents(
    project_id: str,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get all agents in a project"""
    try:
        project = await manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "agents": project.agents,
            "total": len(project.agents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agents for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/workflows")
async def get_project_workflows(
    project_id: str,
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get all workflows in a project"""
    try:
        project = await manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "workflows": project.workflows,
            "total": len(project.workflows)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching workflows for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_project_templates():
    """Get available project templates"""
    try:
        # Discover templates from filesystem
        templates_dir = Path(__file__).parent.parent / "templates" / "project_templates"
        discovered_templates = []

        # Check if templates directory exists
        if templates_dir.exists() and templates_dir.is_dir():
            for template_dir in templates_dir.iterdir():
                if template_dir.is_dir() and (template_dir / "template.json").exists():
                    try:
                        # Load template metadata
                        with open(template_dir / "template.json", 'r') as f:
                            template_meta = json.load(f)
                            template_meta["id"] = template_dir.name
                            discovered_templates.append(template_meta)
                    except Exception as e:
                        logger.warning(f"Failed to load template {template_dir.name}: {e}")

        # Add default templates if no templates found
        if not discovered_templates:
            discovered_templates = [
                {
                    "id": "basic-agent",
                    "name": "Basic Agent",
                    "description": "Simple A2A agent template with basic skills",
                    "type": "agent",
                    "tags": ["basic", "starter"]
                },
                {
                    "id": "data-processor",
                    "name": "Data Processing Agent",
                    "description": "Agent specialized in data processing and transformation",
                    "type": "agent",
                    "tags": ["data", "processing"]
                },
                {
                    "id": "workflow-orchestrator",
                    "name": "Workflow Orchestrator",
                    "description": "BPMN-based workflow orchestration template",
                    "type": "workflow",
                    "tags": ["workflow", "orchestration"]
                }
            ]

        # Combine discovered and default templates
        all_templates = discovered_templates

        # Sort templates by name
        all_templates.sort(key=lambda x: x.get('name', ''))

        return {
            "templates": all_templates,
            "total": len(all_templates),
            "source": "filesystem" if templates_dir.exists() else "defaults"
        }

    except Exception as e:
        logger.error(f"Error fetching templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stats")
async def get_dashboard_stats(
    manager: ProjectDataManager = Depends(get_project_manager)
):
    """Get dashboard statistics"""
    try:
        all_projects = await manager.get_all_projects()

        # Calculate statistics
        stats = {
            "total_projects": len(all_projects),
            "active_projects": len([p for p in all_projects if p.status == ProjectStatus.ACTIVE]),
            "deployed_projects": len([p for p in all_projects if p.deployment_status == DeploymentStatus.DEPLOYED]),
            "total_agents": sum(len(p.agents) for p in all_projects),
            "total_workflows": sum(len(p.workflows) for p in all_projects),
            "project_types": {
                "agent": len([p for p in all_projects if p.project_type == ProjectType.AGENT]),
                "workflow": len([p for p in all_projects if p.project_type == ProjectType.WORKFLOW]),
                "integration": len([p for p in all_projects if p.project_type == ProjectType.INTEGRATION]),
                "template": len([p for p in all_projects if p.project_type == ProjectType.TEMPLATE])
            },
            "recent_projects": [
                p.dict() for p in sorted(all_projects, key=lambda x: x.last_modified, reverse=True)[:5]
            ]
        }

        return stats

    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def initialize_project_structure(project_id: str, template_id: Optional[str] = None):
    """Background task to initialize project structure"""
    try:
        logger.info(f"Initializing project structure for {project_id}")

        # Get workspace path from proper storage configuration
        from ...config.storageConfig import get_workspace_path
        workspace_base = get_workspace_path()
        project_path = workspace_base / project_id

        # Create directory structure
        directories = [
            "src",
            "src/agents",
            "src/workflows",
            "src/skills",
            "src/handlers",
            "tests",
            "tests/unit",
            "tests/integration",
            "docs",
            "config",
            "scripts",
            ".github/workflows"
        ]

        for dir_path in directories:
            (project_path / dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {project_path / dir_path}")

        # Copy template files if template_id provided
        if template_id:
            templates_dir = Path(__file__).parent.parent / "templates" / "project_templates" / template_id
            if templates_dir.exists():
                import shutil
                for item in templates_dir.iterdir():
                    if item.name != "template.json":  # Skip metadata file
                        if item.is_file():
                            shutil.copy2(item, project_path / item.name)
                        elif item.is_dir():
                            shutil.copytree(item, project_path / item.name, dirs_exist_ok=True)
                logger.info(f"Copied template files from {template_id}")

        # Initialize git repository
        try:
            import subprocess
            subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)

            # Create .gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
*.log
.DS_Store
"""
            (project_path / ".gitignore").write_text(gitignore_content)

            # Initial commit
            subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial project structure"],
                         cwd=project_path, check=True, capture_output=True)
            logger.info("Initialized git repository")
        except Exception as e:
            logger.warning(f"Git initialization failed: {e}")

        # Create default configuration files
        config_files = {
            "config/project.yaml": f"""# A2A Project Configuration
project:
  id: {project_id}
  name: {project_id}
  version: "0.1.0"

agents:
  - name: default_agent
    type: basic
    skills: []

workflows:
  - name: default_workflow
    type: sequential
    steps: []
""",
            "README.md": f"""# {project_id}

A2A Agent Project

## Structure

- `src/` - Source code
  - `agents/` - Agent implementations
  - `workflows/` - Workflow definitions
  - `skills/` - Custom skills
  - `handlers/` - Event handlers
- `tests/` - Test files
- `docs/` - Documentation
- `config/` - Configuration files

## Getting Started

1. Install dependencies
2. Configure agents in `config/project.yaml`
3. Run tests with `pytest`
4. Deploy with A2A Developer Portal
""",
            "requirements.txt": """# A2A Agent Dependencies
a2a-sdk>=3.0.0
pydantic>=2.0.0
httpx>=0.24.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
""",
            "src/__init__.py": "",
            "tests/__init__.py": "",
            "tests/test_basic.py": """\"\"\"Basic project tests\"\"\"
import os
import pytest
from pathlib import Path

def test_project_structure():
    \"\"\"Test that project structure is created correctly\"\"\"
    project_root = Path(__file__).parent.parent

    # Test core directories exist
    assert (project_root / "src").exists(), "Source directory should exist"
    assert (project_root / "tests").exists(), "Tests directory should exist"
    assert (project_root / "config").exists(), "Config directory should exist"

    # Test core files exist
    assert (project_root / "package.json").exists(), "package.json should exist"
    assert (project_root / "README.md").exists(), "README.md should exist"

    # Test config files have required content
    if (project_root / "package.json").exists():
        import json
        with open(project_root / "package.json") as f:
            pkg = json.load(f)
            assert "name" in pkg, "package.json should have name field"
            assert "version" in pkg, "package.json should have version field"

    # Test source structure is valid
    src_dir = project_root / "src"
    if src_dir.exists():
        py_files = list(src_dir.glob("**/*.py"))
        js_files = list(src_dir.glob("**/*.js"))
        assert len(py_files) > 0 or len(js_files) > 0, "Project should contain source files"

def test_project_configuration():
    \"\"\"Test that project configuration is valid\"\"\"
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"

    if config_dir.exists():
        config_files = list(config_dir.glob("*.json"))
        for config_file in config_files:
            with open(config_file) as f:
                import json
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in {config_file}")
"""
        }

        for file_path, content in config_files.items():
            file_full_path = project_path / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(content)
            logger.debug(f"Created file: {file_full_path}")

        logger.info(f"Project structure initialized successfully for {project_id}")

    except Exception as e:
        logger.error(f"Error initializing project structure for {project_id}: {e}")
        raise


def initialize_project_api(data_manager: ProjectDataManager):
    """Initialize the project API with data manager"""
    global project_manager
    project_manager = data_manager
    logger.info("Enhanced Project API initialized")

"""
Enhanced Project Data Models for A2A Developer Portal
Connects to real data sources and provides comprehensive project management
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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum
from uuid import uuid4
import logging

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
logger = logging.getLogger(__name__)

Base = declarative_base()


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    TESTING = "testing"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ProjectType(str, Enum):
    """Project type enumeration"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    TEMPLATE = "template"


class DeploymentStatus(str, Enum):
    """Deployment status enumeration"""
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    DEPLOYMENT_FAILED = "deployment_failed"
    UPDATING = "updating"


# SQLAlchemy Models
class ProjectDB(Base):
    """Database model for projects"""
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    project_type = Column(String, nullable=False)
    status = Column(String, default=ProjectStatus.DRAFT)
    deployment_status = Column(String, default=DeploymentStatus.NOT_DEPLOYED)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)
    tags = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    agents = Column(JSON, default=list)
    workflows = Column(JSON, default=list)
    templates = Column(JSON, default=list)
    dependencies = Column(JSON, default=list)
    deployment_config = Column(JSON, default=dict)
    test_results = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)


class AgentDB(Base):
    """Database model for agents"""
    __tablename__ = "agents"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    agent_type = Column(String)
    configuration = Column(JSON, default=dict)
    skills = Column(JSON, default=list)
    handlers = Column(JSON, default=list)
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)
    test_results = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)


class WorkflowDB(Base):
    """Database model for workflows"""
    __tablename__ = "workflows"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    bpmn_xml = Column(Text)
    workflow_config = Column(JSON, default=dict)
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow)
    execution_history = Column(JSON, default=list)


# Pydantic Models
class ProjectMetrics(BaseModel):
    """Project performance metrics"""
    total_agents: int = 0
    total_workflows: int = 0
    test_coverage: float = 0.0
    deployment_success_rate: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ProjectDependency(BaseModel):
    """Project dependency model"""
    name: str
    version: str
    type: str  # "agent", "skill", "library", etc.
    source: str  # "local", "registry", "git", etc.
    required: bool = True


class DeploymentConfig(BaseModel):
    """Deployment configuration"""
    target_environment: str = "development"
    auto_deploy: bool = False
    rollback_enabled: bool = True
    health_check_url: Optional[str] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)


class EnhancedProject(BaseModel):
    """Enhanced project model with real data connections"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    project_type: ProjectType = ProjectType.AGENT
    status: ProjectStatus = ProjectStatus.DRAFT
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = Field(default_factory=list)

    # Enhanced data
    agents: List[Dict[str, Any]] = Field(default_factory=list)
    workflows: List[Dict[str, Any]] = Field(default_factory=list)
    templates: List[str] = Field(default_factory=list)
    dependencies: List[ProjectDependency] = Field(default_factory=list)

    # Configuration and deployment
    deployment_config: DeploymentConfig = Field(default_factory=DeploymentConfig)
    test_results: Dict[str, Any] = Field(default_factory=dict)
    metrics: ProjectMetrics = Field(default_factory=ProjectMetrics)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectDataManager:
    """Enhanced project data manager with real data source connections"""

    def __init__(self, database_url: str = None):
        # Use environment variable or secure default
        self.database_url = database_url or os.getenv('PROJECT_DATABASE_URL', 'sqlite:///./data/a2a_projects.db')

        # Validate database URL for security
        if not self.database_url:
            raise ValueError("Database URL must be configured")

        # Security configuration based on database type
        if self.database_url.startswith('sqlite:'):
            # Secure SQLite configuration
            engine_kwargs = {
                'echo': False,  # Never log SQL in production
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30,
                    'isolation_level': None  # Autocommit mode for better concurrency
                }
            }
        elif self.database_url.startswith(('postgresql:', 'mysql:', 'oracle:')):
            # Secure configuration for production databases
            engine_kwargs = {
                'echo': False,
                'pool_size': 5,
                'max_overflow': 10,
                'pool_pre_ping': True,
                'pool_recycle': 3600,
                'connect_args': {
                    'connect_timeout': 30,
                    'sslmode': 'require' if 'postgresql:' in self.database_url else 'REQUIRED'
                }
            }
        else:
            # Default secure configuration
            engine_kwargs = {'echo': False, 'pool_pre_ping': True}

        self.engine = create_engine(self.database_url, **engine_kwargs)

        # Ensure data directory exists for SQLite
        if self.database_url.startswith('sqlite:'):
            db_path = self.database_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)

        Base.metadata.create_all(bind=self.engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db_session = SessionLocal()

        # In-memory cache for performance
        self.project_cache: Dict[str, EnhancedProject] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_cache_update = datetime.utcnow()

        logger.info(f"ProjectDataManager initialized with database: {database_url}")

    async def get_all_projects(self) -> List[EnhancedProject]:
        """Get all projects from database"""
        try:
            # Check cache first
            if self._is_cache_valid():
                return list(self.project_cache.values())

            # Fetch from database
            db_projects = self.db_session.query(ProjectDB).all()
            projects = []

            for db_project in db_projects:
                project = self._db_to_pydantic(db_project)
                projects.append(project)
                self.project_cache[project.id] = project

            self.last_cache_update = datetime.utcnow()
            return projects

        except Exception as e:
            logger.error(f"Error fetching projects: {e}")
            return []

    async def get_project(self, project_id: str) -> Optional[EnhancedProject]:
        """Get single project by ID"""
        try:
            # Check cache first
            if project_id in self.project_cache and self._is_cache_valid():
                return self.project_cache[project_id]

            # Fetch from database
            db_project = self.db_session.query(ProjectDB).filter(ProjectDB.id == project_id).first()
            if not db_project:
                return None

            project = self._db_to_pydantic(db_project)
            self.project_cache[project_id] = project
            return project

        except Exception as e:
            logger.error(f"Error fetching project {project_id}: {e}")
            return None

    async def create_project(self, project_data: Dict[str, Any]) -> EnhancedProject:
        """Create new project"""
        try:
            project = EnhancedProject(**project_data)

            # Save to database
            db_project = self._pydantic_to_db(project)
            self.db_session.add(db_project)
            self.db_session.commit()

            # Update cache
            self.project_cache[project.id] = project

            logger.info(f"Created project: {project.name} ({project.id})")
            return project

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error creating project: {e}")
            raise

    async def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[EnhancedProject]:
        """Update existing project"""
        try:
            db_project = self.db_session.query(ProjectDB).filter(ProjectDB.id == project_id).first()
            if not db_project:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(db_project, key):
                    setattr(db_project, key, value)

            db_project.last_modified = datetime.utcnow()
            self.db_session.commit()

            # Update cache
            project = self._db_to_pydantic(db_project)
            self.project_cache[project_id] = project

            return project

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error updating project {project_id}: {e}")
            return None

    async def delete_project(self, project_id: str) -> bool:
        """Delete project"""
        try:
            db_project = self.db_session.query(ProjectDB).filter(ProjectDB.id == project_id).first()
            if not db_project:
                return False

            self.db_session.delete(db_project)
            self.db_session.commit()

            # Remove from cache
            self.project_cache.pop(project_id, None)

            logger.info(f"Deleted project: {project_id}")
            return True

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error deleting project {project_id}: {e}")
            return False

    async def get_project_metrics(self, project_id: str) -> Optional[ProjectMetrics]:
        """Get project performance metrics"""
        project = await self.get_project(project_id)
        if not project:
            return None

        # Calculate real-time metrics
        metrics = ProjectMetrics(
            total_agents=len(project.agents),
            total_workflows=len(project.workflows),
            test_coverage=project.test_results.get("coverage", 0.0),
            deployment_success_rate=project.test_results.get("deployment_success_rate", 0.0),
            avg_response_time=project.test_results.get("avg_response_time", 0.0),
            error_rate=project.test_results.get("error_rate", 0.0)
        )

        return metrics

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return datetime.utcnow() - self.last_cache_update < self.cache_ttl

    def _db_to_pydantic(self, db_project: ProjectDB) -> EnhancedProject:
        """Convert database model to Pydantic model"""
        return EnhancedProject(
            id=db_project.id,
            name=db_project.name,
            description=db_project.description or "",
            project_type=ProjectType(db_project.project_type),
            status=ProjectStatus(db_project.status),
            deployment_status=DeploymentStatus(db_project.deployment_status),
            created_at=db_project.created_at,
            last_modified=db_project.last_modified,
            created_by=db_project.created_by or "system",
            tags=db_project.tags or [],
            agents=db_project.agents or [],
            workflows=db_project.workflows or [],
            templates=db_project.templates or [],
            dependencies=[ProjectDependency(**dep) for dep in (db_project.dependencies or [])],
            deployment_config=DeploymentConfig(**(db_project.deployment_config or {})),
            test_results=db_project.test_results or {},
            metrics=ProjectMetrics(**(db_project.performance_metrics or {})),
            metadata=db_project.metadata or {}
        )

    def _pydantic_to_db(self, project: EnhancedProject) -> ProjectDB:
        """Convert Pydantic model to database model"""
        return ProjectDB(
            id=project.id,
            name=project.name,
            description=project.description,
            project_type=project.project_type.value,
            status=project.status.value,
            deployment_status=project.deployment_status.value,
            created_at=project.created_at,
            last_modified=project.last_modified,
            created_by=project.created_by,
            tags=project.tags,
            agents=project.agents,
            workflows=project.workflows,
            templates=project.templates,
            dependencies=[dep.dict() for dep in project.dependencies],
            deployment_config=project.deployment_config.dict(),
            test_results=project.test_results,
            performance_metrics=project.metrics.dict(),
            metadata=project.metadata
        )

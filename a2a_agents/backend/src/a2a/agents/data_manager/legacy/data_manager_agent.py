"""
Data Manager A2A Agent - The actual link to filesystem and databases
This is where data really lives and how CRUD operations happen in the A2A network
"""

import asyncio
import json
import os
import shutil
import gzip
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from uuid import uuid4
import logging
from enum import Enum
import pandas as pd
import hashlib

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.clients.hana_client import create_hana_client
from app.clients.supabase_client import create_supabase_client
from app.ord_registry.service import ORDRegistryService
from app.ord_registry.models import ORDDocument, DublinCoreMetadata, RegistrationRequest
from ..security.smart_contract_trust import initialize_agent_trust, verify_a2a_message, sign_a2a_message, get_trust_contract
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus
from app.services.cache_manager import CacheManager, CacheConfig, cache_result

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ServiceLevel(str, Enum):
    GOLD = "gold"      # Immediate processing, guaranteed storage, full redundancy
    SILVER = "silver"  # Standard processing, reliable storage
    BRONZE = "bronze"  # Best effort, eventual consistency


class DataOperation(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXISTS = "exists"
    ARCHIVE = "archive"
    RESTORE = "restore"


class StorageType(str, Enum):
    FILE = "file"
    HANA = "hana"
    SUPABASE = "supabase"
    DUAL = "dual"  # Both file and database


class MessagePart(BaseModel):
    kind: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    parts: List[MessagePart]
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    signature: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    state: TaskState
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: Optional[A2AMessage] = None
    error: Optional[Dict[str, Any]] = None


class TaskArtifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    parts: List[MessagePart]


class DataRequest(BaseModel):
    """A2A-compliant data request format - uses ORD references, not raw data"""
    operation: DataOperation
    storage_type: StorageType = StorageType.DUAL
    service_level: ServiceLevel = ServiceLevel.SILVER
    
    # A2A Compliant: Reference to data, not data itself
    ord_reference: Optional[str] = None  # ORD registration ID for input data
    schema_reference: Optional[str] = None  # CSN schema reference
    access_pattern: Optional[str] = None  # read_only, read_write, append_only
    
    # Legacy fields for internal operations only (not from A2A messages)
    path: Optional[str] = None  # Internal: File path or table name
    data: Optional[Union[Dict, List[Dict]]] = None  # Internal: Raw data (ONLY for internal ops)
    query: Optional[Dict[str, Any]] = None  # Query filters
    options: Optional[Dict[str, Any]] = None  # Additional options
    
    # Output specification
    output_schema: Optional[str] = None  # Desired output schema
    output_format: Optional[str] = None  # json, csv, parquet


class OperationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class StorageLocation(BaseModel):
    """Precise location of data in storage"""
    storage_type: str  # "file", "hana", "supabase"
    path: Optional[str] = None  # File path
    database: Optional[str] = None  # Database name
    schema: Optional[str] = None  # Schema name
    table: Optional[str] = None  # Table name
    row_count: Optional[int] = None  # Number of rows affected
    primary_key: Optional[Union[str, List[str]]] = None  # Primary key(s) for created/updated records
    connection_string: Optional[str] = None  # Masked connection string
    checksum: Optional[str] = None  # Data integrity checksum
    size_bytes: Optional[int] = None  # Size in bytes


class OperationResult(BaseModel):
    """Result of a single storage operation"""
    model_config = {"extra": "allow"}  # Allow extra fields like 'data'
    
    storage_type: str
    status: OperationStatus
    location: Optional[StorageLocation] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: Optional[float] = None


class DataManagerResponse(BaseModel):
    """A2A-compliant response for data operations - references to data, not data itself"""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    operation: DataOperation
    overall_status: OperationStatus = OperationStatus.FAILED  # Default to failed, updated on success
    primary_result: Optional[OperationResult] = None  # Primary storage result
    failover_result: Optional[OperationResult] = None  # Failover storage result
    
    # A2A Compliant: Data references, not raw data
    ord_output_reference: Optional[str] = None  # ORD registration ID for output data
    schema_reference: Optional[str] = None  # CSN schema of output data
    access_method: Optional[str] = None  # How to access the data (sql_query, file_path, api_endpoint)
    data_location: Optional[Dict[str, Any]] = None  # Where data is stored (not the data itself)
    
    # Internal only - not sent in A2A messages
    data: Optional[Any] = None  # Internal: Raw data for legacy operations
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledgment: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_a2a_message(self, task_id: str, context_id: str) -> A2AMessage:
        """Convert response to A2A-compliant message format - NO raw data transfer"""
        parts = []
        
        # Status part
        status_text = f"Operation {self.operation.value} completed with status: {self.overall_status.value}"
        if self.ord_output_reference:
            status_text += f" - Data available at ORD reference: {self.ord_output_reference}"
        parts.append(MessagePart(kind="text", text=status_text))
        
        # A2A Compliant data reference part
        reference_data = {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp
        }
        
        # Add ORD reference if data was created/processed
        if self.ord_output_reference:
            reference_data["ord_output_reference"] = self.ord_output_reference
            reference_data["schema_reference"] = self.schema_reference
            reference_data["access_method"] = self.access_method
            reference_data["data_location"] = self.data_location
        
        # Add storage location metadata (not data itself)  
        if self.primary_result:
            reference_data["primary_storage"] = {
                "status": self.primary_result.status.value,
                "storage_location": {
                    "type": self.primary_result.location.storage_type if self.primary_result.location else None,
                    "database": self.primary_result.location.database if self.primary_result.location else None,
                    "schema": self.primary_result.location.schema if self.primary_result.location else None,
                    "table": self.primary_result.location.table if self.primary_result.location else None,
                    "path": self.primary_result.location.path if self.primary_result.location else None,
                    "row_count": self.primary_result.location.row_count if self.primary_result.location else None
                },
                "error": self.primary_result.error
            }
            
        if self.failover_result:
            reference_data["failover_storage"] = {
                "status": self.failover_result.status.value,
                "storage_location": {
                    "type": self.failover_result.location.storage_type if self.failover_result.location else None,
                    "database": self.failover_result.location.database if self.failover_result.location else None,
                    "schema": self.failover_result.location.schema if self.failover_result.location else None,
                    "table": self.failover_result.location.table if self.failover_result.location else None,
                    "path": self.failover_result.location.path if self.failover_result.location else None,
                    "row_count": self.failover_result.location.row_count if self.failover_result.location else None
                },
                "error": self.failover_result.error
            }
            
        parts.append(MessagePart(kind="data", data=reference_data))
        
        # Add acknowledgment
        parts.append(MessagePart(kind="data", data={"acknowledgment": self.acknowledgment}))
            
        return A2AMessage(
            role=MessageRole.AGENT,
            parts=parts,
            taskId=task_id,
            contextId=context_id
        )


class AgentCard(BaseModel):
    name: str = "Data Manager A2A Agent"
    description: str = "Microservice for all data operations - CRUD on files and databases"
    url: str
    version: str = "2.0.0"
    protocolVersion: str = "0.2.9"
    provider: Dict[str, str] = {
        "organization": "FinSight CIB",
        "url": "https://finsight-cib.com"
    }
    capabilities: Dict[str, bool] = {
        "streaming": True,
        "pushNotifications": True,
        "stateTransitionHistory": True,
        "crud": True,
        "dualStorage": True,
        "serviceLevels": True,
        "trustIntegration": True,
        "smartContractDelegation": True,
        "aiAdvisor": True,
        "helpSeeking": True,
        "taskTracking": True
    }
    defaultInputModes: List[str] = ["application/json", "text/csv", "application/octet-stream"]
    defaultOutputModes: List[str] = ["application/json", "text/csv", "application/octet-stream"]
    authentication: Dict[str, List[str]] = {
        "schemes": ["Bearer", "Basic"]
    }
    preferredTransport: str = "https"
    additionalInterfaces: Optional[List[Dict[str, Any]]] = None
    skills: List[Dict[str, Any]] = [
        {
            "id": "crud-operations",
            "name": "CRUD Operations",
            "description": "Create, Read, Update, Delete data in files or databases",
            "tags": ["crud", "data", "storage"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "file-management",
            "name": "File Management",
            "description": "Direct filesystem operations - read, write, move, compress",
            "tags": ["file", "filesystem", "storage"],
            "inputModes": ["application/json", "application/octet-stream"],
            "outputModes": ["application/json", "application/octet-stream"]
        },
        {
            "id": "database-management",
            "name": "Database Management",
            "description": "HANA and Supabase operations with SQL support",
            "tags": ["database", "hana", "supabase", "sql"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "dual-storage",
            "name": "Dual Storage Management",
            "description": "Synchronized storage across files and databases",
            "tags": ["dual", "sync", "redundancy"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "service-levels",
            "name": "Service Level Management",
            "description": "Gold/Silver/Bronze service levels with priority queuing",
            "tags": ["sla", "priority", "qos"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "ai-advisor",
            "name": "AI-Powered Help and Guidance",
            "description": "Intelligent advisor for data management troubleshooting using Grok-4",
            "tags": ["ai", "advisor", "help", "grok-4"],
            "inputModes": ["text/plain", "application/json"],
            "outputModes": ["application/json"]
        }
    ]


class DataManagerAgent(AgentHelpSeeker):
    """Data Manager A2A Agent - The single source of truth for all data operations"""
    
    def __init__(self, base_url: str, ord_registry_url: str):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        self.ord_registry_url = ord_registry_url
        
        # Initialize ORD Registry Service for A2A compliance
        if not ord_registry_url:
            raise ValueError("ORD registry URL is required for A2A-compliant Data Manager")
        self.ord_service = ORDRegistryService(base_url=self.ord_registry_url)
        
        # Agent identification (trust identity will be registered via Agent Manager)
        self.agent_id = "data_manager_agent"
        self.agent_name = "DataManagerAgent"
        self.agent_identity = None  # Will be set after Agent Manager registration
        logger.info(f"Data Manager identity pending registration via Agent Manager: {self.agent_id}")
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name="Data Manager Agent"
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "ord_registry_url": self.ord_registry_url,
            "agent_type": "data_manager",
            "timeout": 30.0,
            "retry.max_attempts": 3
        }
        self.initialize_help_action_system(self.agent_id, agent_context)
        
        self.agent_card = AgentCard(url=base_url)
        
        # Initialize Database-backed AI Decision Logger
        from ..core.ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
        
        # Data Manager is special - it stores its own decisions in the same database it manages
        # Use a self-referential URL or direct database access
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=self.agent_id,
            data_manager_url=self.base_url,  # Self-reference for Data Manager
            memory_size=1000,
            learning_threshold=10,
            cache_ttl=300
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI Advisor
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name="Data Manager Agent",
            agent_capabilities={
                "streaming": True,
                "pushNotifications": True,
                "stateTransitionHistory": True,
                "crud": True,
                "dualStorage": True,
                "serviceLevels": True,
                "trustIntegration": True,
                "smartContractDelegation": True,
                "aiAdvisor": True,
                "helpSeeking": True,
                "taskTracking": True
            }
        )
        
        # Initialize message queue with agent-specific configuration
        self.initialize_message_queue(
            agent_id=self.agent_id,
            max_concurrent_processing=6,  # Higher for data operations
            auto_mode_threshold=10,       # Switch to queue after 10 pending
            enable_streaming=True,        # Support real-time processing
            enable_batch_processing=True  # Support batch processing
        )
        
        # Task management with persistence
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.status_updates: Dict[str, List[TaskStatus]] = {}
        self.operation_history: List[Dict[str, Any]] = []  # Track all operations with detailed responses
        
        # Initialize persistent storage for tasks
        self._storage_path = os.getenv("DATA_MANAGER_STORAGE_PATH", "/tmp/data_manager_state")
        os.makedirs(self._storage_path, exist_ok=True)
        
        # Load persisted state
        asyncio.create_task(self._load_persisted_state())
        
        # Data storage configuration
        # Get base path from environment configuration - no hardcoded paths
        self.file_base_path = os.getenv("DATA_MANAGER_BASE_PATH")
        if not self.file_base_path:
            # For testing, create a temporary directory
            import tempfile
            self.file_base_path = tempfile.mkdtemp(prefix="data_manager_")
            logger.warning(f"⚠️ DATA_MANAGER_BASE_PATH not set, using temporary directory: {self.file_base_path}")
        
        self.storage_paths = {
            "raw": os.path.join(self.file_base_path, "raw"),
            "interim": os.path.join(self.file_base_path, "interim"),
            "processed": os.path.join(self.file_base_path, "processed"),
            "archive": os.path.join(self.file_base_path, "archive")
        }
        
        # Ensure directories exist
        for path in self.storage_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # Add knowledge to AI advisor (after file_base_path is set)
        self._initialize_advisor_knowledge()
        
        # Set operational state callback for live status reporting
        self.ai_advisor.set_operational_state_callback(self.get_agent_operational_state)
        
        # Initialize database clients
        self.hana_client = None
        self.supabase_client = None
        self._init_db_clients()
        
        # Service level processors
        self.processors = {
            ServiceLevel.GOLD: self._process_gold,
            ServiceLevel.SILVER: self._process_silver,
            ServiceLevel.BRONZE: self._process_bronze
        }
        
        # Set message queue processor callback after all methods are defined
        if self.message_queue:
            self.message_queue.set_message_processor(self._process_message_core)
        
        # Initialize 3-tier cache manager for BDC Core
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache_config = CacheConfig(
            redis_url=redis_url,
            l1_max_size=10000,
            l1_default_ttl=900,  # 15 minutes
            l2_default_ttl=3600,  # 1 hour
            l3_default_ttl=14400  # 4 hours
        )
        self.cache_manager = CacheManager(cache_config)
        
        # Initialize cache connection asynchronously
        asyncio.create_task(self._initialize_cache())
        
        logger.info("✅ Data Manager Agent v2.0.0 initialized with AI advisor, help-seeking, message queue, and 3-tier caching capabilities")
    
    async def _initialize_cache(self):
        """Initialize cache manager connection"""
        try:
            await self.cache_manager.initialize()
            logger.info("✅ Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache manager: {e}")
            # Continue without caching if Redis is unavailable
    
    def _initialize_advisor_knowledge(self):
        """Initialize AI advisor with agent-specific knowledge"""
        # Add FAQs
        self.ai_advisor.add_faq_item(
            "What does the Data Manager do?",
            "I am the Data Manager Agent. I handle all data storage operations including CRUD operations on files and databases (HANA, Supabase). I provide dual storage with service levels (Gold, Silver, Bronze) and support delegation from other agents for data operations."
        )
        
        self.ai_advisor.add_faq_item(
            "What storage types do you support?",
            "I support file storage (filesystem), HANA database, Supabase database, and dual storage (synchronized across file and database). I can handle CSV, JSON, and binary data with automatic compression and archiving."
        )
        
        self.ai_advisor.add_faq_item(
            "What are your service levels?",
            "I offer three service levels: Gold (immediate processing, guaranteed storage, full redundancy), Silver (standard processing, reliable storage), and Bronze (best effort, eventual consistency). Service levels determine priority and processing guarantees."
        )
        
        # Add common issues
        self.ai_advisor.add_common_issue(
            "database_connection_failed",
            "Failed to connect to HANA or Supabase database",
            "Check database configuration and network connectivity. Verify credentials and service availability. For HANA, ensure the SAP HANA service is running. For Supabase, check API keys and project URL."
        )
        
        self.ai_advisor.add_common_issue(
            "file_permission_denied",
            "Permission denied when accessing file storage paths",
            f"Check file system permissions for the configured storage paths. Ensure the agent has read/write access to {self.file_base_path} and subdirectories. Verify directory ownership and permissions."
        )
        
        self.ai_advisor.add_common_issue(
            "dual_storage_sync_failed",
            "Synchronization between file and database storage failed",
            "Check both file and database connectivity. Verify data integrity and format compatibility. Review service level requirements and retry with appropriate SLA. Consider fallback to single storage mode if sync continues to fail."
        )
        
        # Add help-seeking knowledge
        self.ai_advisor.add_faq_item(
            "Can you ask other agents for help?",
            "Yes! I can actively seek help from other agents when I encounter problems. I know how to contact Agent 0 for data processing guidance, Agent 1 for standardization issues, and the Catalog Manager for metadata problems."
        )
        
        self.ai_advisor.add_common_issue(
            "help_seeking_available",
            "When should I seek help from other agents?",
            "I automatically seek help when I encounter repeated failures (3+ consecutive errors), network connectivity issues, service unavailability, or data integrity problems. I can ask Agent 0 for processing help, Agent 1 for standardization guidance, and others based on the problem type."
        )
    
    def _init_db_clients(self):
        """Initialize database clients for dual storage"""
        # Initialize database clients - fail fast if critical infrastructure is unavailable
        hana_required = os.getenv("DATA_MANAGER_HANA_REQUIRED", "false").lower() == "true"
        supabase_required = os.getenv("DATA_MANAGER_SUPABASE_REQUIRED", "false").lower() == "true"
        
        try:
            self.hana_client = create_hana_client()
            logger.info("✅ HANA client initialized for Data Manager")
        except Exception as e:
            if hana_required:
                raise RuntimeError(f"HANA client initialization failed and is required: {e}")
            logger.warning(f"⚠️ HANA client initialization failed (optional): {e}")
            self.hana_client = None
        
        try:
            self.supabase_client = create_supabase_client()
            logger.info("✅ Supabase client initialized for Data Manager")
        except Exception as e:
            if supabase_required:
                raise RuntimeError(f"Supabase client initialization failed and is required: {e}")
            logger.warning(f"⚠️ Supabase client initialization failed (optional): {e}")
            self.supabase_client = None
        
        # Ensure at least one database client is available
        if not self.hana_client and not self.supabase_client:
            raise RuntimeError("No database clients available. Data Manager requires at least one working database connection.")
    
    def _is_advisor_request(self, message: A2AMessage) -> bool:
        """Check if message is requesting AI advisor help"""
        for part in message.parts:
            if part.kind == "text" and part.text:
                text_lower = part.text.lower()
                if any(word in text_lower for word in ["help", "advisor", "question", "how", "what", "explain", "troubleshoot"]):
                    return True
            elif part.kind == "data" and part.data:
                if "advisor_request" in part.data or "help_request" in part.data:
                    return True
        return False
    
    async def _handle_advisor_request(self, message: A2AMessage) -> A2AMessage:
        """Handle AI advisor help requests with decision logging"""
        
        # Extract question from message
        question = ""
        for part in message.parts:
            if part.kind == "text" and part.text:
                question = part.text
                break
            elif part.kind == "data" and part.data and isinstance(part.data, dict):
                question = part.data.get("question", str(part.data))
                break
        
        # Context for decision logging
        context = {
            "context_id": message.contextId,
            "asking_agent_id": getattr(message, 'from_agent_id', 'unknown'),
            "message_parts": len(message.parts),
            "processing_stage": "data_manager_advisor"
        }
        
        # Log the AI decision request
        from ..core.ai_decision_logger import DecisionType, OutcomeStatus
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            question=question,
            ai_response={},  # Will update after getting response
            context=context
        )
        
        try:
            asking_agent_id = getattr(message, 'from_agent_id', None)
            start_time = asyncio.get_event_loop().time()
            
            advisor_response = await self.ai_advisor.process_a2a_help_message(
                [part.dict() for part in message.parts],
                asking_agent_id
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Update decision with response details
            if decision_id in self.ai_decision_logger._decision_cache:
                decision = self.ai_decision_logger._decision_cache[decision_id]
                decision.ai_response = advisor_response
                decision.response_time = response_time
            
            # Determine success based on response
            has_answer = (
                isinstance(advisor_response, dict) and 
                advisor_response.get('answer') and 
                len(advisor_response['answer']) > 10
            )
            
            # Log outcome
            await self.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.SUCCESS if has_answer else OutcomeStatus.PARTIAL_SUCCESS,
                success_metrics={
                    "response_length": len(str(advisor_response)),
                    "has_answer": has_answer,
                    "response_time": response_time,
                    "advisor_available": True
                }
            )
            
            # Create A2A response message
            response = A2AMessage(
                role=MessageRole.AGENT,
                taskId=message.taskId,
                contextId=message.contextId,
                parts=[{
                    "kind": "data",
                    "data": {
                        "message_type": "advisor_response",
                        "advisor_response": advisor_response,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "decision_metadata": {
                            "decision_id": decision_id,
                            "database_backed": True,
                            "learning_active": True
                        }
                    }
                }]
            )
            
            # Sign response
            response.signature = sign_a2a_message(response.model_dump())
            return response
            
        except Exception as e:
            # Log failure outcome
            await self.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.FAILURE,
                failure_reason=str(e),
                success_metrics={"exception_occurred": True, "error_type": type(e).__name__}
            )
            
            logger.error(f"❌ Error handling advisor request: {e}")
            error_response = A2AMessage(
                role=MessageRole.AGENT,
                taskId=message.taskId,
                contextId=message.contextId,
                parts=[{
                    "kind": "data",
                    "data": {
                        "message_type": "advisor_error",
                        "error": str(e),
                        "agent_id": self.agent_id,
                        "decision_metadata": {
                            "decision_id": decision_id,
                            "database_backed": True
                        }
                    }
                }]
            )
            error_response.signature = sign_a2a_message(error_response.model_dump())
            return error_response
    
    async def _handle_error_with_help_seeking(
        self, 
        error: Exception, 
        operation_name: str,
        original_operation: Optional[Callable] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle errors by seeking help from other agents using trust-restricted system"""
        
        try:
            # Map operation to problem type
            problem_type_mapping = {
                "database_operation": "data_storage",
                "file_operation": "file_operations", 
                "crud_operation": "data_storage",
                "hana_operation": "database_connectivity",
                "supabase_operation": "database_connectivity"
            }
            
            problem_type = problem_type_mapping.get(operation_name, "data_storage")
            
            # Add task to tracker
            task_id = f"error_handling_{operation_name}_{int(datetime.utcnow().timestamp())}"
            self.task_tracker.add_task(
                task_id=task_id,
                description=f"Handle {operation_name} error using help-seeking",
                priority=TaskPriority.HIGH,
                context={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "operation": operation_name
                }
            )
            
            # Seek help and act on response
            help_result = await self.seek_help_and_act(
                problem_type=problem_type,
                error=error,
                original_operation=original_operation,
                context=context or {},
                urgency="high"
            )
            
            if help_result["success"] and help_result.get("resolved_issue"):
                self.task_tracker.complete_task(task_id, {
                    "result": "Error resolved through help-seeking",
                    "help_actions_taken": help_result.get("actions_executed", 0),
                    "resolution_method": "trust_restricted_help"
                })
                
                logger.info(f"✅ Error resolved through trust-restricted help for {operation_name}")
                return {"success": True, "resolved": True, "method": "help_seeking"}
            else:
                self.task_tracker.fail_task(task_id, {
                    "reason": "Help-seeking failed to resolve error",
                    "help_sought": help_result.get("help_sought", False),
                    "help_received": help_result.get("help_received", False)
                })
                
                logger.warning(f"⚠️ Help-seeking failed to resolve {operation_name} error")
                return {"success": False, "resolved": False, "method": "help_seeking"}
                
        except Exception as help_error:
            logger.error(f"❌ Error in help-seeking process: {help_error}")
            return {"success": False, "resolved": False, "error": str(help_error)}
    
    async def process_message(
        self, 
        message: A2AMessage,
        context_id: str = None,
        priority: str = "medium",
        processing_mode: str = "auto"
    ) -> Dict[str, Any]:
        """Process A2A message with queue support (streaming or batched)"""
        
        # Use context_id from message if not provided
        if not context_id:
            context_id = message.contextId or str(uuid4())
        
        if not self.message_queue:
            # Fallback to direct processing if queue not initialized
            result = await self._process_message_core(message, context_id)
            # Convert A2AMessage result to dict for consistency
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.dict()
            return result
        
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
        
        # Create or retrieve task
        task_id = message.taskId or str(uuid4())
        context_id = message.contextId
        
        if task_id not in self.tasks:
            self.tasks[task_id] = {
                "created": datetime.utcnow(),
                "context_id": context_id,
                "messages": [],
                "status": TaskState.PENDING,
                "artifacts": []
            }
            self.status_updates[task_id] = []
            
            # Persist new task immediately
            asyncio.create_task(self._persist_state())
        
        # Store message
        self.tasks[task_id]["messages"].append(message)
        
        # Extract data request from message
        data_request = await self._extract_data_request(message)
        
        # Execute based on service level
        asyncio.create_task(self._execute_with_service_level(task_id, data_request))
        
        # Create response
        response = A2AMessage(
            role=MessageRole.AGENT,
            taskId=task_id,
            contextId=context_id,
            parts=[{
                "kind": "text",
                "text": f"Data operation '{data_request.operation}' accepted with {data_request.service_level} service level"
            }]
        )
        
        # Sign response
        response.signature = sign_a2a_message(response.model_dump())
        
        return response
    
    async def _extract_data_request(self, message: A2AMessage) -> DataRequest:
        """Extract data request from A2A message"""
        for part in message.parts:
            if part.kind == "data" and part.data:
                # Direct data request
                if "operation" in part.data:
                    return DataRequest(**part.data)
                
                # ORD reference - fetch from registry
                if "ord_reference" in part.data:
                    return DataRequest(
                        operation=DataOperation.READ,
                        storage_type=StorageType.DUAL,
                        query={"ord_reference": part.data["ord_reference"]}
                    )
        
        # Default read operation
        return DataRequest(operation=DataOperation.READ)
    
    async def _execute_with_service_level(self, task_id: str, request: DataRequest):
        """Execute request based on service level"""
        try:
            await self._update_status(task_id, TaskState.WORKING)
            
            # Route to appropriate processor
            processor = self.processors.get(request.service_level, self._process_silver)
            response = await processor(task_id, request)
            
            # Convert response to A2A message
            response_message = response.to_a2a_message(task_id, context_id)
            
            # Create artifact with detailed location info
            artifact = TaskArtifact(
                name=f"{request.operation} Operation Result",
                description=f"Result of {request.operation} with {request.service_level} service",
                parts=response_message.parts
            )
            self.tasks[task_id]["artifacts"].append(artifact)
            
            # Store the response for task status
            self.tasks[task_id]["response"] = response
            
            # Track operation in history for audit and acknowledgment
            operation_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task_id,
                "context_id": context_id,
                "request_id": response.request_id,
                "operation": response.operation.value,
                "service_level": request.service_level.value,
                "overall_status": response.overall_status.value,
                "primary_location": response.primary_result.location.model_dump() if response.primary_result and response.primary_result.location else None,
                "failover_location": response.failover_result.location.model_dump() if response.failover_result and response.failover_result.location else None,
                "acknowledgment": response.acknowledgment
            }
            self.operation_history.append(operation_record)
            
            await self._update_status(
                task_id, 
                TaskState.COMPLETED,
                f"Operation {request.operation} completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Operation execution error: {e}")
            await self._update_status(
                task_id,
                TaskState.FAILED,
                error={
                    "code": "OPERATION_ERROR",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
    
    async def _process_gold(self, task_id: str, request: DataRequest) -> DataManagerResponse:
        """Gold service - immediate processing with full redundancy"""
        logger.info(f"🥇 Processing GOLD request: {request.operation}")
        
        # Always use dual storage for gold
        if request.storage_type == StorageType.FILE:
            request.storage_type = StorageType.DUAL
        
        return await self._execute_operation(request)
    
    async def _process_silver(self, task_id: str, request: DataRequest) -> DataManagerResponse:
        """Silver service - standard processing"""
        logger.info(f"🥈 Processing SILVER request: {request.operation}")
        return await self._execute_operation(request)
    
    async def _process_bronze(self, task_id: str, request: DataRequest) -> DataManagerResponse:
        """Bronze service - best effort"""
        logger.info(f"🥉 Processing BRONZE request: {request.operation}")
        
        # Add small delay for bronze
        await asyncio.sleep(0.1)
        
        return await self._execute_operation(request)
    
    async def _execute_operation(self, request: DataRequest) -> DataManagerResponse:
        """Execute the actual data operation with primary/failover support"""
        operations = {
            DataOperation.CREATE: self._create,
            DataOperation.READ: self._read,
            DataOperation.UPDATE: self._update,
            DataOperation.DELETE: self._delete,
            DataOperation.LIST: self._list,
            DataOperation.EXISTS: self._exists,
            DataOperation.ARCHIVE: self._archive,
            DataOperation.RESTORE: self._restore
        }
        
        operation_func = operations.get(request.operation)
        if not operation_func:
            raise ValueError(f"Unknown operation: {request.operation}")
        
        # Execute operation and get response
        response = await operation_func(request)
        
        # Record operation in acknowledgment
        response.acknowledgment = {
            "request_received": datetime.utcnow().isoformat(),
            "service_level": request.service_level.value,
            "storage_type_requested": request.storage_type.value,
            "agent_id": self.agent_id,
            "trust_contract": get_trust_contract().contract_id if get_trust_contract() else None
        }
        
        return response
    
    async def _create(self, request: DataRequest) -> DataManagerResponse:
        """Create data in storage with primary/failover support - A2A compliant"""
        response = DataManagerResponse(
            operation=DataOperation.CREATE,
            overall_status=OperationStatus.FAILED  # Start as failed, update on success
        )
        start_time = datetime.utcnow()
        
        # A2A Compliance: Handle ORD reference input
        actual_data = None
        if request.ord_reference:
            # A2A compliant: get data using ORD reference
            try:
                actual_data = await self._get_data_by_ord_reference(request.ord_reference, request.query)
                logger.info(f"✅ Retrieved data from ORD reference: {request.ord_reference}")
            except Exception as e:
                response.overall_status = OperationStatus.FAILED
                response.primary_result = OperationResult(
                    storage_type="ord_resolution",
                    status=OperationStatus.FAILED,
                    error=f"Failed to resolve ORD reference {request.ord_reference}: {str(e)}"
                )
                return response
        elif request.data:
            # Legacy: direct data input (internal operations only)
            actual_data = request.data
            logger.warning("⚠️ Using direct data input - not A2A compliant")
        else:
            response.overall_status = OperationStatus.FAILED
            response.primary_result = OperationResult(
                storage_type="input_validation",
                status=OperationStatus.FAILED,
                error="Either ord_reference or data must be provided"
            )
            return response
        
        # Determine primary and failover based on storage type
        primary_storage = None
        failover_storage = None
        
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE
            failover_storage = StorageType.HANA  # Default failover to HANA
        elif request.storage_type in [StorageType.FILE, StorageType.HANA, StorageType.SUPABASE]:
            primary_storage = request.storage_type
        
        # Execute primary storage operation
        if primary_storage == StorageType.FILE and request.path:
            try:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                # Write data
                if request.path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(request.data, f, indent=2)
                elif request.path.endswith('.csv'):
                    df = pd.DataFrame(request.data if isinstance(request.data, list) else [request.data])
                    df.to_csv(file_path, index=False)
                else:
                    # Raw write
                    with open(file_path, 'w') as f:
                        f.write(str(request.data))
                
                # Create location info
                location = StorageLocation(
                    storage_type="file",
                    path=file_path,
                    size_bytes=os.path.getsize(file_path),
                    checksum=self._calculate_checksum(file_path)
                )
                
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.SUCCESS,
                    location=location,
                    duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                )
                
            except Exception as e:
                logger.error(f"File creation failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # If primary failed and we have dual storage, try failover immediately
                if request.storage_type == StorageType.DUAL:
                    failover_storage = StorageType.HANA
        
        # Execute database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (failover_storage and response.primary_result and response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                
                # Create a database-specific request
                db_request = DataRequest(
                    operation=request.operation,
                    storage_type=failover_storage or primary_storage,  # Use specific DB type
                    service_level=request.service_level,
                    data=actual_data,  # Use the resolved actual_data
                    query=request.query,
                    options=request.options
                )
                
                db_result = await self._create_in_database_v2(db_request)
                
                if failover_storage:
                    response.failover_result = db_result
                else:
                    response.primary_result = db_result
                    
                if response.failover_result:
                    response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                
            except Exception as e:
                logger.error(f"Database creation failed: {e}")
                result = OperationResult(
                    storage_type=str(primary_storage or failover_storage),
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if failover_storage:
                    response.failover_result = result
                else:
                    response.primary_result = result
        
        # For DUAL storage, also try database as secondary even if file succeeded
        if (request.storage_type == StorageType.DUAL and 
            response.primary_result and 
            response.primary_result.status == OperationStatus.SUCCESS and 
            not response.failover_result):
            
            try:
                db_start = datetime.utcnow()
                
                # Create a database-specific request for secondary storage
                db_request = DataRequest(
                    operation=request.operation,
                    storage_type=StorageType.HANA,  # Default to HANA for dual storage
                    service_level=request.service_level,
                    data=actual_data,  # Use the resolved actual_data
                    query=request.query,
                    options=request.options
                )
                
                response.failover_result = await self._create_in_database_v2(db_request)
                response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Secondary database storage failed: {e}")
                response.failover_result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
        
        # Determine overall status first
        if response.primary_result and response.primary_result.status == OperationStatus.SUCCESS:
            if response.failover_result and response.failover_result.status == OperationStatus.FAILED:
                response.overall_status = OperationStatus.PARTIAL_SUCCESS
            else:
                response.overall_status = OperationStatus.SUCCESS
        elif response.failover_result and response.failover_result.status == OperationStatus.SUCCESS:
            response.overall_status = OperationStatus.PARTIAL_SUCCESS
        else:
            response.overall_status = OperationStatus.FAILED
        
        # A2A Compliance: Register output data in ORD instead of returning raw data
        if response.overall_status in [OperationStatus.SUCCESS, OperationStatus.PARTIAL_SUCCESS]:
            try:
                # Register the created data in ORD registry
                ord_output_reference = await self._register_data_in_ord(
                    data=actual_data,
                    operation_type="create",
                    storage_location=response.primary_result.location if response.primary_result else response.failover_result.location,
                    context=request.query or {}
                )
                response.ord_output_reference = ord_output_reference
                logger.info(f"✅ A2A Compliant: Data registered in ORD with reference {ord_output_reference}")
                
                # Remove raw data from response for A2A compliance
                response.data = None
                
            except Exception as ord_error:
                logger.error(f"❌ Failed to register data in ORD: {ord_error}")
                # Don't fail the operation, but log the compliance issue
                response.metadata["ord_registration_error"] = str(ord_error)
            
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    async def _read(self, request: DataRequest) -> DataManagerResponse:
        """Read data from storage with primary/failover support"""
        
        # Generate cache key for read operations
        cache_key_parts = [
            str(request.storage_type.value),
            request.path or "no_path",
            str(hash(str(request.query or {})))
        ]
        cache_key = ":".join(cache_key_parts)
        
        # Check cache first
        cached_response = await self.cache_manager.get("read", cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for read operation: {cache_key}")
            return cached_response
        
        response = DataManagerResponse(
            operation=DataOperation.read,
            overall_status=OperationStatus.FAILED
        )
        start_time = datetime.utcnow()
        
        # Determine primary storage
        primary_storage = None
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE if request.path else StorageType.HANA
        else:
            primary_storage = request.storage_type
            
        # Try primary storage first
        if primary_storage == StorageType.FILE and request.path:
            try:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                # Read data based on file type
                data = None
                if request.path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                elif request.path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    data = df.to_dict('records')
                else:
                    with open(file_path, 'r') as f:
                        data = f.read()
                
                # Create location info
                location = StorageLocation(
                    storage_type="file",
                    path=file_path,
                    size_bytes=os.path.getsize(file_path),
                    checksum=self._calculate_checksum(file_path)
                )
                
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.SUCCESS,
                    location=location,
                    duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                )
                response.data = data
                response.overall_status = OperationStatus.SUCCESS
                
            except Exception as e:
                logger.error(f"File read failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # Try failover to database if dual storage
                if request.storage_type == StorageType.DUAL:
                    primary_storage = StorageType.HANA
        
        # Try database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (request.storage_type == StorageType.DUAL and 
             response.primary_result and 
             response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                db_result = await self._read_from_database_v2(request)
                
                if response.primary_result and response.primary_result.status == OperationStatus.FAILED:
                    response.failover_result = db_result
                    if db_result.status == OperationStatus.SUCCESS:
                        # Extract data from custom attribute
                        response.data = getattr(db_result, 'data', None)
                        response.overall_status = OperationStatus.PARTIAL_SUCCESS
                else:
                    response.primary_result = db_result
                    if db_result.status == OperationStatus.SUCCESS:
                        # Extract data from custom attribute
                        response.data = getattr(db_result, 'data', None)
                        response.overall_status = OperationStatus.SUCCESS
                        
                if db_result.status == OperationStatus.SUCCESS:
                    db_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                    
            except Exception as e:
                logger.error(f"Database read failed: {e}")
                result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if response.primary_result and response.primary_result.status == OperationStatus.FAILED:
                    response.failover_result = result
                    response.overall_status = OperationStatus.FAILED
                else:
                    response.primary_result = result
                    response.overall_status = OperationStatus.FAILED
        
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Cache successful read operations (15 minutes TTL)
        if response.overall_status == OperationStatus.SUCCESS:
            await self.cache_manager.set("read", cache_key, response, ttl=900, level=2)
            logger.debug(f"Cached read operation result: {cache_key}")
        
        return response
    
    async def _update(self, request: DataRequest) -> DataManagerResponse:
        """Update existing data with primary/failover support - A2A compliant"""
        response = DataManagerResponse(operation=DataOperation.UPDATE)
        start_time = datetime.utcnow()
        
        # A2A Compliance: Handle ORD reference input
        actual_data = None
        if request.ord_reference:
            # A2A compliant: get data using ORD reference
            try:
                actual_data = await self._get_data_by_ord_reference(request.ord_reference, request.query)
                logger.info(f"✅ Retrieved data from ORD reference: {request.ord_reference}")
            except Exception as e:
                response.overall_status = OperationStatus.FAILED
                response.primary_result = OperationResult(
                    storage_type="ord_resolution",
                    status=OperationStatus.FAILED,
                    error=f"Failed to resolve ORD reference {request.ord_reference}: {str(e)}"
                )
                return response
        elif request.data:
            # Legacy: direct data input (internal operations only)
            actual_data = request.data
            logger.warning("⚠️ Using direct data input - not A2A compliant")
        else:
            response.overall_status = OperationStatus.FAILED
            response.primary_result = OperationResult(
                storage_type="input_validation",
                status=OperationStatus.FAILED,
                error="Either ord_reference or data must be provided"
            )
            return response
        
        # Determine primary and failover based on storage type
        primary_storage = None
        failover_storage = None
        
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE
            failover_storage = StorageType.HANA
        elif request.storage_type in [StorageType.FILE, StorageType.HANA, StorageType.SUPABASE]:
            primary_storage = request.storage_type
        
        # Execute primary storage operation
        if primary_storage == StorageType.FILE and request.path:
            try:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                # Backup original
                backup_path = file_path + f".backup.{int(datetime.utcnow().timestamp())}"
                if os.path.exists(file_path):
                    shutil.copy2(file_path, backup_path)
                
                # Update file using A2A compliant data source
                if request.path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(actual_data, f, indent=2)
                elif request.path.endswith('.csv'):
                    df = pd.DataFrame(actual_data if isinstance(actual_data, list) else [actual_data])
                    df.to_csv(file_path, index=False)
                else:
                    with open(file_path, 'w') as f:
                        f.write(str(actual_data))
                
                # Create location info
                location = StorageLocation(
                    storage_type="file",
                    path=file_path,
                    size_bytes=os.path.getsize(file_path),
                    checksum=self._calculate_checksum(file_path)
                )
                location.backup_path = backup_path  # Custom field for updates
                
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.SUCCESS,
                    location=location,
                    duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                )
                
            except Exception as e:
                logger.error(f"File update failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # Try failover if dual storage
                if request.storage_type == StorageType.DUAL:
                    failover_storage = StorageType.HANA
        
        # Execute database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (failover_storage and response.primary_result and response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                db_result = await self._update_in_database_v2(request)
                
                if failover_storage:
                    response.failover_result = db_result
                else:
                    response.primary_result = db_result
                    
                db_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                
            except Exception as e:
                logger.error(f"Database update failed: {e}")
                result = OperationResult(
                    storage_type=str(primary_storage or failover_storage),
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if failover_storage:
                    response.failover_result = result
                else:
                    response.primary_result = result
        
        # For DUAL storage, also try database as secondary even if file succeeded
        if (request.storage_type == StorageType.DUAL and 
            response.primary_result and 
            response.primary_result.status == OperationStatus.SUCCESS and 
            not response.failover_result):
            
            try:
                db_start = datetime.utcnow()
                response.failover_result = await self._update_in_database_v2(request)
                response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Secondary database update failed: {e}")
                response.failover_result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
        
        # Determine overall status
        if response.primary_result and response.primary_result.status == OperationStatus.SUCCESS:
            if response.failover_result and response.failover_result.status == OperationStatus.FAILED:
                response.overall_status = OperationStatus.PARTIAL_SUCCESS
            else:
                response.overall_status = OperationStatus.SUCCESS
        elif response.failover_result and response.failover_result.status == OperationStatus.SUCCESS:
            response.overall_status = OperationStatus.PARTIAL_SUCCESS
        else:
            response.overall_status = OperationStatus.FAILED
            
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    async def _delete(self, request: DataRequest) -> DataManagerResponse:
        """Delete data with audit trail and primary/failover support"""
        response = DataManagerResponse(operation=DataOperation.DELETE)
        start_time = datetime.utcnow()
        
        # Determine primary and failover based on storage type
        primary_storage = None
        failover_storage = None
        
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE
            failover_storage = StorageType.HANA
        elif request.storage_type in [StorageType.FILE, StorageType.HANA, StorageType.SUPABASE]:
            primary_storage = request.storage_type
        
        # Execute primary storage operation
        if primary_storage == StorageType.FILE and request.path:
            try:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                if os.path.exists(file_path):
                    # Archive instead of delete for safety
                    archive_name = f"{os.path.basename(file_path)}.{int(datetime.utcnow().timestamp())}"
                    archive_path = os.path.join(self.storage_paths["archive"], archive_name)
                    shutil.move(file_path, archive_path)
                    
                    # Create location info
                    location = StorageLocation(
                        storage_type="file",
                        path=file_path,  # Original path
                        size_bytes=os.path.getsize(archive_path),  # Size of archived file
                        checksum=self._calculate_checksum(archive_path)
                    )
                    location.archived_path = archive_path  # Custom field for deletes
                    
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.SUCCESS,
                        location=location,
                        duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                    )
                else:
                    # File doesn't exist
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.FAILED,
                        error=f"File not found: {file_path}"
                    )
                
            except Exception as e:
                logger.error(f"File delete failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # Try failover if dual storage
                if request.storage_type == StorageType.DUAL:
                    failover_storage = StorageType.HANA
        
        # Execute database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (failover_storage and response.primary_result and response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                db_result = await self._delete_from_database_v2(request)
                
                if failover_storage:
                    response.failover_result = db_result
                else:
                    response.primary_result = db_result
                    
                db_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                
            except Exception as e:
                logger.error(f"Database delete failed: {e}")
                result = OperationResult(
                    storage_type=str(primary_storage or failover_storage),
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if failover_storage:
                    response.failover_result = result
                else:
                    response.primary_result = result
        
        # For DUAL storage, also try database as secondary even if file succeeded
        if (request.storage_type == StorageType.DUAL and 
            response.primary_result and 
            response.primary_result.status == OperationStatus.SUCCESS and 
            not response.failover_result):
            
            try:
                db_start = datetime.utcnow()
                response.failover_result = await self._delete_from_database_v2(request)
                response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Secondary database delete failed: {e}")
                response.failover_result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
        
        # Determine overall status
        if response.primary_result and response.primary_result.status == OperationStatus.SUCCESS:
            if response.failover_result and response.failover_result.status == OperationStatus.FAILED:
                response.overall_status = OperationStatus.PARTIAL_SUCCESS
            else:
                response.overall_status = OperationStatus.SUCCESS
        elif response.failover_result and response.failover_result.status == OperationStatus.SUCCESS:
            response.overall_status = OperationStatus.PARTIAL_SUCCESS
        else:
            response.overall_status = OperationStatus.FAILED
            
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    async def _list(self, request: DataRequest) -> DataManagerResponse:
        """List available data with primary/failover support"""
        response = DataManagerResponse(operation=DataOperation.LIST)
        start_time = datetime.utcnow()
        
        items = []
        
        # Determine primary and failover based on storage type
        primary_storage = None
        failover_storage = None
        
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE
            failover_storage = StorageType.HANA
        elif request.storage_type in [StorageType.FILE, StorageType.HANA, StorageType.SUPABASE]:
            primary_storage = request.storage_type
        
        # Execute primary storage operation
        if primary_storage == StorageType.FILE:
            try:
                file_start = datetime.utcnow()
                base_path = request.path or ""
                search_path = self._resolve_path(base_path)
                
                file_items = []
                if os.path.isdir(search_path):
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_items.append({
                                "type": "file",
                                "path": os.path.relpath(file_path, self.file_base_path),
                                "size": os.path.getsize(file_path),
                                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            })
                
                items.extend(file_items)
                
                # Create location info
                location = StorageLocation(
                    storage_type="file",
                    path=search_path,
                    row_count=len(file_items)
                )
                
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.SUCCESS,
                    location=location,
                    duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                )
                
            except Exception as e:
                logger.error(f"File listing failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # Try failover if dual storage
                if request.storage_type == StorageType.DUAL:
                    failover_storage = StorageType.HANA
        
        # Execute database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (failover_storage and response.primary_result and response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                db_result = await self._list_in_database_v2(request)
                
                if failover_storage:
                    response.failover_result = db_result
                    if db_result.status == OperationStatus.SUCCESS:
                        # Extract data from custom attribute
                        db_items = getattr(db_result, 'data', [])
                        items.extend(db_items)
                else:
                    response.primary_result = db_result
                    if db_result.status == OperationStatus.SUCCESS:
                        # Extract data from custom attribute
                        db_items = getattr(db_result, 'data', [])
                        items.extend(db_items)
                        
                db_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                
            except Exception as e:
                logger.error(f"Database listing failed: {e}")
                result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if failover_storage:
                    response.failover_result = result
                else:
                    response.primary_result = result
        
        # For DUAL storage, also try database as secondary even if file succeeded
        if (request.storage_type == StorageType.DUAL and 
            response.primary_result and 
            response.primary_result.status == OperationStatus.SUCCESS and 
            not response.failover_result):
            
            try:
                db_start = datetime.utcnow()
                response.failover_result = await self._list_in_database_v2(request)
                if response.failover_result.status == OperationStatus.SUCCESS:
                    db_items = getattr(response.failover_result, 'data', [])
                    items.extend(db_items)
                response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Secondary database listing failed: {e}")
                response.failover_result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
        
        # Set the data in response
        response.data = {"items": items}
        
        # Determine overall status
        if response.primary_result and response.primary_result.status == OperationStatus.SUCCESS:
            if response.failover_result and response.failover_result.status == OperationStatus.FAILED:
                response.overall_status = OperationStatus.PARTIAL_SUCCESS
            else:
                response.overall_status = OperationStatus.SUCCESS
        elif response.failover_result and response.failover_result.status == OperationStatus.SUCCESS:
            response.overall_status = OperationStatus.PARTIAL_SUCCESS
        else:
            response.overall_status = OperationStatus.FAILED
            
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        response.metadata["total_items"] = len(items)
        
        return response
    
    async def _exists(self, request: DataRequest) -> DataManagerResponse:
        """Check if data exists with primary/failover support"""
        response = DataManagerResponse(operation=DataOperation.EXISTS)
        start_time = datetime.utcnow()
        
        exists_data = {"exists": False, "locations": []}
        
        # Determine primary and failover based on storage type
        primary_storage = None
        failover_storage = None
        
        if request.storage_type == StorageType.DUAL:
            primary_storage = StorageType.FILE
            failover_storage = StorageType.HANA
        elif request.storage_type in [StorageType.FILE, StorageType.HANA, StorageType.SUPABASE]:
            primary_storage = request.storage_type
        
        # Execute primary storage operation
        if primary_storage == StorageType.FILE and request.path:
            try:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                file_exists = os.path.exists(file_path)
                if file_exists:
                    exists_data["exists"] = True
                    exists_data["locations"].append({
                        "source": "file",
                        "path": file_path,
                        "type": "file",
                        "size": os.path.getsize(file_path)
                    })
                
                # Create location info
                location = StorageLocation(
                    storage_type="file",
                    path=file_path,
                    size_bytes=os.path.getsize(file_path) if file_exists else 0
                )
                
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.SUCCESS,
                    location=location,
                    duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                )
                
            except Exception as e:
                logger.error(f"File exists check failed: {e}")
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                # Try failover if dual storage
                if request.storage_type == StorageType.DUAL:
                    failover_storage = StorageType.HANA
        
        # Execute database storage (as primary or failover)
        if (primary_storage in [StorageType.HANA, StorageType.SUPABASE] or 
            (failover_storage and response.primary_result and response.primary_result.status == OperationStatus.FAILED)):
            
            try:
                db_start = datetime.utcnow()
                db_result = await self._exists_in_database_v2(request)
                
                if failover_storage:
                    response.failover_result = db_result
                else:
                    response.primary_result = db_result
                    
                # Extract existence data from custom attribute
                if db_result.status == OperationStatus.SUCCESS:
                    db_exists_data = getattr(db_result, 'data', {})
                    if db_exists_data.get("exists"):
                        exists_data["exists"] = True
                        exists_data["locations"].extend(db_exists_data.get("locations", []))
                        
                db_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
                
            except Exception as e:
                logger.error(f"Database exists check failed: {e}")
                result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
                
                if failover_storage:
                    response.failover_result = result
                else:
                    response.primary_result = result
        
        # For DUAL storage, also try database as secondary even if file succeeded
        if (request.storage_type == StorageType.DUAL and 
            response.primary_result and 
            response.primary_result.status == OperationStatus.SUCCESS and 
            not response.failover_result):
            
            try:
                db_start = datetime.utcnow()
                response.failover_result = await self._exists_in_database_v2(request)
                if response.failover_result.status == OperationStatus.SUCCESS:
                    db_exists_data = getattr(response.failover_result, 'data', {})
                    if db_exists_data.get("exists"):
                        exists_data["exists"] = True
                        exists_data["locations"].extend(db_exists_data.get("locations", []))
                response.failover_result.duration_ms = (datetime.utcnow() - db_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Secondary database exists check failed: {e}")
                response.failover_result = OperationResult(
                    storage_type="database",
                    status=OperationStatus.FAILED,
                    error=str(e)
                )
        
        # Set the data in response
        response.data = exists_data
        
        # Determine overall status
        if response.primary_result and response.primary_result.status == OperationStatus.SUCCESS:
            if response.failover_result and response.failover_result.status == OperationStatus.FAILED:
                response.overall_status = OperationStatus.PARTIAL_SUCCESS
            else:
                response.overall_status = OperationStatus.SUCCESS
        elif response.failover_result and response.failover_result.status == OperationStatus.SUCCESS:
            response.overall_status = OperationStatus.PARTIAL_SUCCESS
        else:
            response.overall_status = OperationStatus.FAILED
            
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    async def _register_data_in_ord(self, data: Any, schema_reference: str, storage_location: StorageLocation, 
                                   operation_type: str = "data_storage") -> str:
        """Register data in ORD registry and return registration ID - A2A compliance"""
        try:
            # Create ORD document with Dublin Core metadata
            ord_doc = ORDDocument(
                title=f"Data Manager Storage - {operation_type}",
                description=f"Data stored by Data Manager Agent in {storage_location.storage_type}",
                dublin_core=DublinCoreMetadata(
                    title=f"Data Manager {operation_type.title()} Result",
                    creator="FinSight Data Manager Agent",
                    type="Dataset",
                    format=storage_location.storage_type,
                    date=datetime.utcnow().isoformat(),
                    identifier=str(uuid4())
                ),
                # CSN schema reference
                schema_reference=schema_reference,
                # Storage location metadata (not raw data)
                access_information={
                    "storage_type": storage_location.storage_type,
                    "database": storage_location.database,
                    "schema": storage_location.schema,
                    "table": storage_location.table,
                    "path": storage_location.path,
                    "connection_string": storage_location.connection_string,
                    "row_count": storage_location.row_count,
                    "primary_key": storage_location.primary_key,
                    "checksum": storage_location.checksum,
                    "size_bytes": storage_location.size_bytes
                },
                data_lineage={
                    "source_agent": self.agent_id,
                    "operation": operation_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Register in ORD
            registration = await self.ord_service.register_ord_document(ord_doc, self.agent_id)
            
            if registration:
                logger.info(f"✅ Data registered in ORD: {registration.registration_id}")
                return registration.registration_id
            else:
                raise Exception("ORD registration failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to register data in ORD: {e}")
            raise
    
    async def _resolve_ord_reference(self, ord_reference: str) -> Dict[str, Any]:
        """Resolve ORD reference to get data location and schema - A2A compliance"""
        try:
            registration = await self.ord_service.get_registration(ord_reference)
            
            if not registration:
                raise ValueError(f"ORD reference not found: {ord_reference}")
            
            # Extract access information from ORD document
            access_info = registration.ord_document.access_information or {}
            schema_ref = registration.ord_document.schema_reference
            
            return {
                "storage_type": access_info.get("storage_type"),
                "database": access_info.get("database"),
                "schema": access_info.get("schema"),
                "table": access_info.get("table"),
                "path": access_info.get("path"),
                "connection_string": access_info.get("connection_string"),
                "row_count": access_info.get("row_count"),
                "checksum": access_info.get("checksum"),
                "schema_reference": schema_ref,
                "ord_metadata": registration.ord_document.dublin_core.model_dump() if registration.ord_document.dublin_core else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve ORD reference {ord_reference}: {e}")
            raise
    
    async def _get_data_by_ord_reference(self, ord_reference: str, query: Optional[Dict[str, Any]] = None) -> Any:
        """Internal method to retrieve actual data using ORD reference - for internal operations only"""
        # This is internal - raw data should NEVER be sent via A2A messages
        
        # Check cache first for ORD resolution
        cache_key = f"ord_data:{ord_reference}"
        if query:
            cache_key += f":{hash(str(query))}"
        
        # Try to get from cache
        cached_result = await self.cache_manager.get("ord", cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for ORD reference: {ord_reference}")
            return cached_result
        
        # Cache miss - resolve ORD reference
        access_info = await self._resolve_ord_reference(ord_reference)
        
        storage_type = access_info.get("storage_type")
        
        result = None
        
        if storage_type == "file":
            file_path = access_info.get("path")
            if file_path and os.path.exists(file_path):
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    result = df.to_dict('records')
                else:
                    with open(file_path, 'r') as f:
                        result = f.read()
                        
        elif storage_type in ["hana", "supabase"]:
            table = access_info.get("table")
            if table:
                if storage_type == "hana" and self.hana_client:
                    result = self.hana_client.query(table, query or {})
                elif storage_type == "supabase" and self.supabase_client:
                    query_builder = self.supabase_client.table(table).select("*")
                    if query:
                        for key, value in query.items():
                            query_builder = query_builder.eq(key, value)
                    response = await query_builder.execute()
                    result = response.data
        
        if result is None:
            raise ValueError(f"Cannot retrieve data from storage type: {storage_type}")
        
        # Cache the result for future use (1 hour TTL)
        await self.cache_manager.set("ord", cache_key, result, ttl=3600, level=2)
        logger.debug(f"Cached ORD reference result: {ord_reference}")
        
        return result
    
    async def _exists_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Check if data exists in database with detailed location tracking"""
        # Determine which database to use
        db_type = None
        if request.storage_type == StorageType.HANA or (request.storage_type == StorageType.DUAL and self.hana_client):
            db_type = "hana"
        elif request.storage_type == StorageType.SUPABASE or (request.storage_type == StorageType.DUAL and self.supabase_client):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            exists_data = {"exists": False, "locations": []}
            
            if db_type == "hana" and self.hana_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                schema_name = "PUBLIC"
                
                # Check if table exists
                check_sql = """
                    SELECT COUNT(*) as count 
                    FROM SYS.TABLES 
                    WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                    AND TABLE_NAME = ?
                """
                result = self.hana_client.execute_query(check_sql, [table_name])
                
                if result and result[0].get("count", 0) > 0:
                    exists_data["exists"] = True
                    
                    # Get record count
                    count_sql = f"SELECT COUNT(*) as record_count FROM {table_name}"
                    count_result = self.hana_client.execute_query(count_sql)
                    record_count = count_result[0].get("record_count", 0) if count_result else 0
                    
                    exists_data["locations"].append({
                        "source": "hana",
                        "table": table_name,
                        "type": "table",
                        "count": record_count,
                        "schema": schema_name
                    })
                
                # Get connection info
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema=schema_name,
                    table=table_name,
                    row_count=record_count if exists_data["exists"] else 0,
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                result.data = exists_data  # Attach data as custom attribute
                return result
                
            elif db_type == "supabase" and self.supabase_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                
                # Check if table exists by trying to access it
                try:
                    response = await self.supabase_client.table(table_name).select("*", count="exact").limit(0).execute()
                    exists_data["exists"] = True
                    exists_data["locations"].append({
                        "source": "supabase",
                        "table": table_name,
                        "type": "table",
                        "count": response.count
                    })
                except Exception:
                    # Table doesn't exist or no access
                    exists_data["exists"] = False
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    table=table_name,
                    row_count=response.count if exists_data["exists"] else 0,
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                result.data = exists_data  # Attach data as custom attribute
                return result
                
        except Exception as e:
            logger.error(f"Database exists check error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _archive(self, request: DataRequest) -> DataManagerResponse:
        """Archive data with compression"""
        response = DataManagerResponse(operation=DataOperation.ARCHIVE)
        start_time = datetime.utcnow()
        
        try:
            if request.path:
                file_start = datetime.utcnow()
                file_path = self._resolve_path(request.path)
                
                if os.path.exists(file_path):
                    # Compress and move to archive
                    archive_name = f"{os.path.basename(file_path)}.{int(datetime.utcnow().timestamp())}.gz"
                    archive_path = os.path.join(self.storage_paths["archive"], archive_name)
                    
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(archive_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original if requested
                    if request.options and request.options.get("remove_original", False):
                        os.remove(file_path)
                    
                    # Create location info
                    location = StorageLocation(
                        storage_type="file",
                        path=archive_path,  # Archive location
                        size_bytes=os.path.getsize(archive_path),
                        checksum=self._calculate_checksum(archive_path)
                    )
                    location.original_path = file_path  # Custom field for archives
                    location.compressed = True
                    
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.SUCCESS,
                        location=location,
                        duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                    )
                    
                    response.data = {
                        "original": file_path,
                        "archive": archive_path,
                        "compressed_size": os.path.getsize(archive_path),
                        "removed_original": request.options and request.options.get("remove_original", False)
                    }
                    response.overall_status = OperationStatus.SUCCESS
                    
                else:
                    # File doesn't exist
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.FAILED,
                        error=f"File not found: {file_path}"
                    )
                    response.overall_status = OperationStatus.FAILED
            else:
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error="Path is required for archive operation"
                )
                response.overall_status = OperationStatus.FAILED
                
        except Exception as e:
            logger.error(f"Archive operation failed: {e}")
            response.primary_result = OperationResult(
                storage_type="file",
                status=OperationStatus.FAILED,
                error=str(e)
            )
            response.overall_status = OperationStatus.FAILED
        
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    async def _restore(self, request: DataRequest) -> DataManagerResponse:
        """Restore archived data"""
        response = DataManagerResponse(operation=DataOperation.RESTORE)
        start_time = datetime.utcnow()
        
        try:
            if request.path:
                file_start = datetime.utcnow()
                archive_path = os.path.join(self.storage_paths["archive"], request.path)
                
                if os.path.exists(archive_path):
                    # Determine restore location
                    if archive_path.endswith('.gz'):
                        restore_name = os.path.basename(archive_path)[:-3]  # Remove .gz
                        restore_path = self._resolve_path(restore_name.rsplit('.', 1)[0])
                        
                        # Decompress
                        with gzip.open(archive_path, 'rb') as f_in:
                            with open(restore_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        # Simple restore
                        restore_path = self._resolve_path(os.path.basename(archive_path).rsplit('.', 1)[0])
                        shutil.copy2(archive_path, restore_path)
                    
                    # Create location info
                    location = StorageLocation(
                        storage_type="file",
                        path=restore_path,  # Restored location
                        size_bytes=os.path.getsize(restore_path),
                        checksum=self._calculate_checksum(restore_path)
                    )
                    location.archive_path = archive_path  # Custom field for restores
                    location.decompressed = archive_path.endswith('.gz')
                    
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.SUCCESS,
                        location=location,
                        duration_ms=(datetime.utcnow() - file_start).total_seconds() * 1000
                    )
                    
                    response.data = {
                        "archive": archive_path,
                        "restored_to": restore_path,
                        "decompressed": archive_path.endswith('.gz')
                    }
                    response.overall_status = OperationStatus.SUCCESS
                    
                else:
                    # Archive doesn't exist
                    response.primary_result = OperationResult(
                        storage_type="file",
                        status=OperationStatus.FAILED,
                        error=f"Archive not found: {archive_path}"
                    )
                    response.overall_status = OperationStatus.FAILED
            else:
                response.primary_result = OperationResult(
                    storage_type="file",
                    status=OperationStatus.FAILED,
                    error="Path is required for restore operation"
                )
                response.overall_status = OperationStatus.FAILED
                
        except Exception as e:
            logger.error(f"Restore operation failed: {e}")
            response.primary_result = OperationResult(
                storage_type="file",
                status=OperationStatus.FAILED,
                error=str(e)
            )
            response.overall_status = OperationStatus.FAILED
        
        response.metadata["total_duration_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return response
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative path to absolute path within data directory"""
        # Remove any leading slashes
        path = path.lstrip('/')
        
        # Determine base directory from path segments
        if path.startswith('raw/'):
            base = self.storage_paths["raw"]
            path = path[4:]
        elif path.startswith('interim/'):
            base = self.storage_paths["interim"]
            path = path[8:]
        elif path.startswith('processed/'):
            base = self.storage_paths["processed"]
            path = path[10:]
        elif path.startswith('archive/'):
            base = self.storage_paths["archive"]
            path = path[8:]
        else:
            # Default to raw
            base = self.storage_paths["raw"]
        
        full_path = os.path.join(base, path)
        
        # Security: ensure path doesn't escape data directory
        if not full_path.startswith(self.file_base_path):
            raise ValueError(f"Invalid path: {path}")
        
        return full_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    async def _create_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Create data in database with detailed location tracking"""
        # Determine which database to use
        db_type = None
        if request.storage_type == StorageType.HANA or (request.storage_type == StorageType.DUAL and self.hana_client):
            db_type = "hana"
        elif request.storage_type == StorageType.SUPABASE or (request.storage_type == StorageType.DUAL and self.supabase_client):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            if db_type == "hana" and self.hana_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                schema_name = "PUBLIC"  # Default schema, could be made configurable
                
                # Ensure table exists
                await self._ensure_hana_table(table_name, request.data)
                
                # Insert data and track primary keys
                inserted_keys = []
                if isinstance(request.data, list):
                    for record in request.data:
                        # Assuming ID field exists or is auto-generated
                        result = self.hana_client.insert(table_name, record)
                        if result and 'id' in result:
                            inserted_keys.append(result['id'])
                else:
                    result = self.hana_client.insert(table_name, request.data)
                    if result and 'id' in result:
                        inserted_keys.append(result['id'])
                
                # Get connection info (masked)
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema=schema_name,
                    table=table_name,
                    row_count=len(request.data) if isinstance(request.data, list) else 1,
                    primary_key=inserted_keys if inserted_keys else None,
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
            elif db_type == "supabase" and self.supabase_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                
                # Insert data
                if isinstance(request.data, list):
                    response = await self.supabase_client.table(table_name).insert(request.data).execute()
                else:
                    response = await self.supabase_client.table(table_name).insert(request.data).execute()
                
                # Extract primary keys from response
                inserted_keys = []
                if response.data:
                    for record in response.data:
                        if 'id' in record:
                            inserted_keys.append(record['id'])
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    table=table_name,
                    row_count=len(response.data) if response.data else 0,
                    primary_key=inserted_keys if inserted_keys else None,
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
        except Exception as e:
            logger.error(f"Database create error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _create_in_database(self, request: DataRequest) -> Dict[str, Any]:
        """Create data in database"""
        results = {}
        
        # HANA
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                # Ensure table exists
                await self._ensure_hana_table(table_name, request.data)
                
                # Insert data
                if isinstance(request.data, list):
                    self.hana_client.bulk_insert(table_name, request.data)
                else:
                    self.hana_client.insert(table_name, request.data)
                
                results["hana"] = {
                    "table": table_name,
                    "records": len(request.data) if isinstance(request.data, list) else 1
                }
            except Exception as e:
                logger.error(f"HANA create error: {e}")
                if request.storage_type == StorageType.HANA:
                    raise
        
        # Supabase
        if self.supabase_client and request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                # Insert data
                response = self.supabase_client.upsert(
                    table_name,
                    request.data if isinstance(request.data, list) else [request.data]
                )
                
                results["supabase"] = {
                    "table": table_name,
                    "records": len(response.data) if response.data else 0
                }
            except Exception as e:
                logger.error(f"Supabase create error: {e}")
                if request.storage_type == StorageType.SUPABASE:
                    raise
        
        return results
    
    async def _read_from_database_v2(self, request: DataRequest) -> OperationResult:
        """Read data from database with location tracking"""
        # Determine which database to use
        db_type = None
        if self.hana_client and (request.storage_type == StorageType.HANA or request.storage_type == StorageType.DUAL):
            db_type = "hana"
        elif self.supabase_client and (request.storage_type == StorageType.SUPABASE or request.storage_type == StorageType.DUAL):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            if db_type == "hana" and self.hana_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                schema_name = "PUBLIC"
                
                # Build query
                if request.query:
                    # Use provided query filters
                    results = self.hana_client.query(table_name, request.query)
                else:
                    # Read all
                    results = self.hana_client.query(table_name, {})
                
                # Get connection info
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema=schema_name,
                    table=table_name,
                    row_count=len(results) if results else 0,
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                # Store data in the result (will be transferred to response.data)
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                # Attach data as a custom attribute
                result.data = results
                return result
                
            elif db_type == "supabase" and self.supabase_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                
                # Build query
                query = self.supabase_client.table(table_name).select("*")
                
                if request.query:
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                
                response = await query.execute()
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    table=table_name,
                    row_count=len(response.data) if response.data else 0,
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                # Attach data as custom attribute
                result.data = response.data
                return result
                
        except Exception as e:
            logger.error(f"Database read error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _read_from_database(self, request: DataRequest) -> Dict[str, Any]:
        """Read data from database"""
        results = {"data": None, "metadata": {}}
        
        # HANA
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                # Build query
                if request.query and request.query.get("sql"):
                    # Direct SQL
                    data = self.hana_client.execute_query(request.query["sql"])
                else:
                    # Simple select
                    where_clause = ""
                    if request.query:
                        conditions = [f"{k} = '{v}'" for k, v in request.query.items()]
                        where_clause = f"WHERE {' AND '.join(conditions)}"
                    
                    sql = f"SELECT * FROM {table_name} {where_clause}"
                    data = self.hana_client.execute_query(sql)
                
                if data:
                    results["data"] = data
                    results["metadata"]["source"] = "hana"
                    results["metadata"]["table"] = table_name
                    results["metadata"]["count"] = len(data)
                    
            except Exception as e:
                logger.error(f"HANA read error: {e}")
        
        # Supabase fallback
        if not results["data"] and self.supabase_client and \
           request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                # Build query
                query = self.supabase_client.table(table_name).select("*")
                
                if request.query:
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                
                response = query.execute()
                
                if response.data:
                    results["data"] = response.data
                    results["metadata"]["source"] = "supabase"
                    results["metadata"]["table"] = table_name
                    results["metadata"]["count"] = len(response.data)
                    
            except Exception as e:
                logger.error(f"Supabase read error: {e}")
        
        return results
    
    async def _update_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Update data in database with detailed location tracking"""
        # Determine which database to use
        db_type = None
        if request.storage_type == StorageType.HANA or (request.storage_type == StorageType.DUAL and self.hana_client):
            db_type = "hana"
        elif request.storage_type == StorageType.SUPABASE or (request.storage_type == StorageType.DUAL and self.supabase_client):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            if db_type == "hana" and self.hana_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                schema_name = "PUBLIC"
                
                # Update data
                updated_count = 0
                if request.query:
                    # Update with specific conditions
                    updated_count = self.hana_client.update(table_name, request.data, request.query)
                else:
                    # Update all records (dangerous but allowed)
                    updated_count = self.hana_client.update(table_name, request.data, {})
                
                # Get connection info
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema=schema_name,
                    table=table_name,
                    row_count=updated_count,
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
            elif db_type == "supabase" and self.supabase_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                
                # Update data
                query = self.supabase_client.table(table_name)
                
                if request.query:
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                
                response = await query.update(request.data).execute()
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    table=table_name,
                    row_count=len(response.data) if response.data else 0,
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
        except Exception as e:
            logger.error(f"Database update error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _update_in_database(self, request: DataRequest) -> Dict[str, Any]:
        """Update data in database"""
        results = {"updated": 0, "metadata": {}}
        
        # HANA update operations
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                # Build SET clause from request.data
                if not request.data:
                    raise ValueError("No data provided for update operation")
                
                set_clauses = []
                values = []
                for key, value in request.data.items():
                    if key != 'id':  # Don't update ID column
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                # Build WHERE clause from request.query
                where_clauses = []
                if request.query:
                    for key, value in request.query.items():
                        where_clauses.append(f"{key} = ?")
                        values.append(value)
                else:
                    raise ValueError("No WHERE conditions provided for update")
                
                set_clause = ", ".join(set_clauses)
                where_clause = " AND ".join(where_clauses)
                sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
                
                updated_count = self.hana_client.execute_update(sql, values)
                results["updated"] = updated_count
                results["metadata"]["source"] = "hana"
                results["metadata"]["table"] = table_name
                
            except Exception as e:
                logger.error(f"HANA update error: {e}")
                if request.storage_type == StorageType.HANA:
                    raise
        
        # Supabase update operations
        if self.supabase_client and request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                if not request.data:
                    raise ValueError("No data provided for update operation")
                if not request.query:
                    raise ValueError("No WHERE conditions provided for update")
                
                # Build Supabase update query
                query = self.supabase_client.table(table_name).update(request.data)
                
                # Apply WHERE conditions
                for key, value in request.query.items():
                    query = query.eq(key, value)
                
                response = query.execute()
                
                if response.data:
                    results["updated"] = len(response.data)
                    results["metadata"]["source"] = "supabase"
                    results["metadata"]["table"] = table_name
                    
            except Exception as e:
                logger.error(f"Supabase update error: {e}")
                if request.storage_type == StorageType.SUPABASE:
                    raise
        
        return results
    
    async def _delete_from_database_v2(self, request: DataRequest) -> OperationResult:
        """Delete data from database with detailed location tracking"""
        # Determine which database to use
        db_type = None
        if request.storage_type == StorageType.HANA or (request.storage_type == StorageType.DUAL and self.hana_client):
            db_type = "hana"
        elif request.storage_type == StorageType.SUPABASE or (request.storage_type == StorageType.DUAL and self.supabase_client):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            if db_type == "hana" and self.hana_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                schema_name = "PUBLIC"
                
                # Soft delete preferred - add deleted_at timestamp
                deleted_count = 0
                if request.query:
                    # Delete with specific conditions
                    deleted_count = self.hana_client.soft_delete(table_name, request.query)
                else:
                    # Delete all records (dangerous but tracked)
                    deleted_count = self.hana_client.soft_delete(table_name, {})
                
                # Get connection info
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema=schema_name,
                    table=table_name,
                    row_count=deleted_count,
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
            elif db_type == "supabase" and self.supabase_client:
                if not request.path:
                    raise ValueError("Table path is required for database operations")
                    
                table_name = request.path
                
                # Soft delete - update with deleted_at timestamp
                query = self.supabase_client.table(table_name)
                
                if request.query:
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                
                # Soft delete by updating deleted_at field
                soft_delete_data = {"deleted_at": datetime.utcnow().isoformat()}
                response = await query.update(soft_delete_data).execute()
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    table=table_name,
                    row_count=len(response.data) if response.data else 0,
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                
        except Exception as e:
            logger.error(f"Database delete error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _delete_from_database(self, request: DataRequest) -> Dict[str, Any]:
        """Delete data from database (soft delete preferred)"""
        results = {"deleted": 0, "metadata": {}, "soft_delete": True}
        
        # HANA delete operations
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                if not request.query:
                    raise ValueError("No WHERE conditions provided for delete operation")
                
                # Check if table supports soft delete (has deleted_at column)
                supports_soft_delete = self.hana_client.check_column_exists(table_name, "deleted_at")
                
                if supports_soft_delete and request.context.get("hard_delete") != True:
                    # Soft delete - update deleted_at timestamp
                    where_clauses = []
                    values = [datetime.utcnow().isoformat()]  # deleted_at value
                    
                    for key, value in request.query.items():
                        where_clauses.append(f"{key} = ?")
                        values.append(value)
                    
                    where_clause = " AND ".join(where_clauses)
                    sql = f"UPDATE {table_name} SET deleted_at = ? WHERE {where_clause} AND deleted_at IS NULL"
                    
                    deleted_count = self.hana_client.execute_update(sql, values)
                    results["soft_delete"] = True
                else:
                    # Hard delete
                    where_clauses = []
                    values = []
                    
                    for key, value in request.query.items():
                        where_clauses.append(f"{key} = ?")
                        values.append(value)
                    
                    where_clause = " AND ".join(where_clauses)
                    sql = f"DELETE FROM {table_name} WHERE {where_clause}"
                    
                    deleted_count = self.hana_client.execute_update(sql, values)
                    results["soft_delete"] = False
                
                results["deleted"] = deleted_count
                results["metadata"]["source"] = "hana"
                results["metadata"]["table"] = table_name
                
            except Exception as e:
                logger.error(f"HANA delete error: {e}")
                if request.storage_type == StorageType.HANA:
                    raise
        
        # Supabase delete operations
        if self.supabase_client and request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                if not request.query:
                    raise ValueError("No WHERE conditions provided for delete operation")
                
                # Check if soft delete is supported and requested
                if request.context.get("hard_delete") != True:
                    # Try soft delete first
                    try:
                        query = self.supabase_client.table(table_name).update({
                            "deleted_at": datetime.utcnow().isoformat()
                        })
                        
                        # Apply WHERE conditions
                        for key, value in request.query.items():
                            query = query.eq(key, value)
                        
                        # Only update records that aren't already soft deleted
                        query = query.is_("deleted_at", "null")
                        
                        response = query.execute()
                        
                        if response.data:
                            results["deleted"] = len(response.data)
                            results["soft_delete"] = True
                        
                    except Exception:
                        # Fall back to hard delete if soft delete fails
                        query = self.supabase_client.table(table_name).delete()
                        
                        for key, value in request.query.items():
                            query = query.eq(key, value)
                        
                        response = query.execute()
                        results["deleted"] = len(response.data) if response.data else 0
                        results["soft_delete"] = False
                else:
                    # Hard delete requested
                    query = self.supabase_client.table(table_name).delete()
                    
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                    
                    response = query.execute()
                    results["deleted"] = len(response.data) if response.data else 0
                    results["soft_delete"] = False
                
                results["metadata"]["source"] = "supabase"
                results["metadata"]["table"] = table_name
                
            except Exception as e:
                logger.error(f"Supabase delete error: {e}")
                if request.storage_type == StorageType.SUPABASE:
                    raise
        
        return results
    
    async def _list_in_database_v2(self, request: DataRequest) -> OperationResult:
        """List data in database with detailed location tracking"""
        # Determine which database to use
        db_type = None
        if request.storage_type == StorageType.HANA or (request.storage_type == StorageType.DUAL and self.hana_client):
            db_type = "hana"
        elif request.storage_type == StorageType.SUPABASE or (request.storage_type == StorageType.DUAL and self.supabase_client):
            db_type = "supabase"
            
        if not db_type:
            return OperationResult(
                storage_type="database",
                status=OperationStatus.FAILED,
                error="No database client available"
            )
            
        try:
            items = []
            
            if db_type == "hana" and self.hana_client:
                # Query HANA system tables for user tables
                sql = """
                    SELECT TABLE_NAME, TABLE_TYPE, RECORD_COUNT, CREATE_TIME
                    FROM SYS.TABLES 
                    WHERE SCHEMA_NAME = CURRENT_SCHEMA
                    AND TABLE_TYPE = 'TABLE'
                    ORDER BY TABLE_NAME
                """
                
                tables = self.hana_client.execute_query(sql)
                
                for table in tables:
                    items.append({
                        "name": table.get("TABLE_NAME"),
                        "type": "table",
                        "source": "hana",
                        "record_count": table.get("RECORD_COUNT", 0),
                        "created": table.get("CREATE_TIME"),
                        "schema": "current_schema"
                    })
                
                # Get connection info
                conn_info = self.hana_client.get_connection_info() if hasattr(self.hana_client, 'get_connection_info') else {}
                
                location = StorageLocation(
                    storage_type="hana",
                    database=conn_info.get('database', 'HANA'),
                    schema="PUBLIC",
                    row_count=len(items),
                    connection_string=f"hana://{conn_info.get('host', 'localhost')}:{conn_info.get('port', 30015)}/{conn_info.get('database', 'HANA')}"
                )
                
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                result.data = items  # Attach data as custom attribute
                return result
                
            elif db_type == "supabase" and self.supabase_client:
                # Get table information from information_schema
                try:
                    response = await self.supabase_client.rpc('get_table_list').execute()
                    
                    if response.data:
                        for table_info in response.data:
                            items.append({
                                "name": table_info.get("table_name"),
                                "type": "table", 
                                "source": "supabase",
                                "record_count": table_info.get("row_count", 0),
                                "schema": table_info.get("table_schema", "public")
                            })
                    else:
                        # Fallback - try common table names
                        common_tables = ["data_manager_records", "standardized_data", "ord_documents"]
                        for table_name in common_tables:
                            try:
                                count_response = await self.supabase_client.table(table_name).select("*", count="exact").limit(0).execute()
                                items.append({
                                    "name": table_name,
                                    "type": "table",
                                    "source": "supabase",
                                    "record_count": count_response.count,
                                    "schema": "public"
                                })
                            except Exception:
                                # Table doesn't exist, skip
                                pass
                                
                except Exception as e:
                    logger.warning(f"Could not get Supabase table list via RPC: {e}")
                    # Continue with empty list
                
                location = StorageLocation(
                    storage_type="supabase",
                    database="supabase",
                    schema="public",
                    row_count=len(items),
                    connection_string=f"postgresql://supabase.{self.supabase_client.supabase_url}"
                )
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=location
                )
                result.data = items  # Attach data as custom attribute
                return result
                
        except Exception as e:
            logger.error(f"Database list error in {db_type}: {e}")
            return OperationResult(
                storage_type=db_type,
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _list_in_database(self, request: DataRequest) -> List[Dict[str, Any]]:
        """List database tables/collections"""
        items = []
        
        # HANA table listing
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                # Query HANA system tables for user tables
                sql = """
                    SELECT TABLE_NAME, TABLE_TYPE, RECORD_COUNT, CREATE_TIME
                    FROM SYS.TABLES 
                    WHERE SCHEMA_NAME = CURRENT_SCHEMA
                    AND TABLE_TYPE = 'TABLE'
                    ORDER BY TABLE_NAME
                """
                
                tables = self.hana_client.execute_query(sql)
                
                for table in tables:
                    items.append({
                        "name": table.get("TABLE_NAME"),
                        "type": "table",
                        "source": "hana",
                        "record_count": table.get("RECORD_COUNT", 0),
                        "created": table.get("CREATE_TIME"),
                        "schema": "current_schema"
                    })
                    
            except Exception as e:
                logger.error(f"HANA list tables error: {e}")
        
        # Supabase table listing
        if self.supabase_client and request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                # Get table information from information_schema
                response = self.supabase_client.rpc('get_table_list').execute()
                
                if response.data:
                    for table_info in response.data:
                        items.append({
                            "name": table_info.get("table_name"),
                            "type": "table", 
                            "source": "supabase",
                            "record_count": table_info.get("row_count", 0),
                            "schema": table_info.get("table_schema", "public")
                        })
                else:
                    # Fallback - try common table names
                    common_tables = ["data_manager_records", "standardized_data", "ord_documents"]
                    for table_name in common_tables:
                        try:
                            count_response = self.supabase_client.table(table_name).select("*", count="exact").limit(0).execute()
                            items.append({
                                "name": table_name,
                                "type": "table",
                                "source": "supabase", 
                                "record_count": count_response.count,
                                "schema": "public"
                            })
                        except:
                            # Table doesn't exist or no access
                            continue
                    
            except Exception as e:
                logger.error(f"Supabase list tables error: {e}")
        
        # Remove duplicates based on table name
        seen_names = set()
        unique_items = []
        for item in items:
            if item["name"] not in seen_names:
                seen_names.add(item["name"])
                unique_items.append(item)
        
        return unique_items
    
    async def _exists_in_database(self, request: DataRequest) -> Dict[str, Any]:
        """Check if data exists in database"""
        results = {"exists": False, "locations": []}
        
        # HANA existence check
        if self.hana_client and request.storage_type in [StorageType.HANA, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                if request.query:
                    # Check for specific records
                    where_clauses = []
                    values = []
                    for key, value in request.query.items():
                        where_clauses.append(f"{key} = ?")
                        values.append(value)
                    
                    where_clause = " AND ".join(where_clauses)
                    sql = f"SELECT COUNT(*) as count FROM {table_name} WHERE {where_clause}"
                    
                    result = self.hana_client.execute_query(sql, values)
                    if result and result[0].get("count", 0) > 0:
                        results["exists"] = True
                        results["locations"].append({
                            "source": "hana",
                            "table": table_name,
                            "count": result[0]["count"]
                        })
                else:
                    # Check if table exists
                    sql = """
                        SELECT COUNT(*) as count 
                        FROM SYS.TABLES 
                        WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                        AND TABLE_NAME = ?
                    """
                    result = self.hana_client.execute_query(sql, [table_name])
                    if result and result[0].get("count", 0) > 0:
                        results["exists"] = True
                        results["locations"].append({
                            "source": "hana",
                            "table": table_name,
                            "type": "table"
                        })
                        
            except Exception as e:
                logger.error(f"HANA exists check error: {e}")
        
        # Supabase existence check
        if self.supabase_client and request.storage_type in [StorageType.SUPABASE, StorageType.DUAL]:
            try:
                if not request.path:
                    raise ValueError("Table path is required for database operations. Cannot use default table name as it may cause data corruption.")
                table_name = request.path
                
                if request.query:
                    # Check for specific records
                    query = self.supabase_client.table(table_name).select("*", count="exact").limit(0)
                    
                    for key, value in request.query.items():
                        query = query.eq(key, value)
                    
                    response = query.execute()
                    if response.count and response.count > 0:
                        results["exists"] = True
                        results["locations"].append({
                            "source": "supabase",
                            "table": table_name,
                            "count": response.count
                        })
                else:
                    # Check if table exists by trying to access it
                    try:
                        response = self.supabase_client.table(table_name).select("*", count="exact").limit(0).execute()
                        results["exists"] = True
                        results["locations"].append({
                            "source": "supabase",
                            "table": table_name,
                            "type": "table",
                            "count": response.count
                        })
                    except Exception:
                        # Table doesn't exist or no access
                        pass
                        
            except Exception as e:
                logger.error(f"Supabase exists check error: {e}")
        
        return results
    
    async def _ensure_hana_table(self, table_name: str, sample_data: Union[Dict, List[Dict]]):
        """Ensure HANA table exists based on sample data"""
        if not self.hana_client:
            return
            
        try:
            # Check if table already exists
            check_sql = """
                SELECT COUNT(*) as count 
                FROM SYS.TABLES 
                WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                AND TABLE_NAME = ?
            """
            result = self.hana_client.execute_query(check_sql, [table_name])
            if result and result[0].get("count", 0) > 0:
                logger.info(f"Table {table_name} already exists")
                return
            
            # Analyze sample data to determine column types
            if isinstance(sample_data, list) and sample_data:
                sample_record = sample_data[0]
            elif isinstance(sample_data, dict):
                sample_record = sample_data
            else:
                raise ValueError("Invalid sample data format")
            
            # Map Python types to HANA types
            type_mapping = {
                str: "NVARCHAR(5000)",
                int: "BIGINT", 
                float: "DOUBLE",
                bool: "BOOLEAN",
                datetime: "TIMESTAMP"
            }
            
            columns = []
            for key, value in sample_record.items():
                if key == "id":
                    columns.append("id NVARCHAR(36) PRIMARY KEY")
                elif key.endswith("_at") or key.endswith("_time"):
                    columns.append(f"{key} TIMESTAMP")
                elif isinstance(value, (int, float)) and key.endswith("_count"):
                    columns.append(f"{key} BIGINT")
                elif isinstance(value, bool):
                    columns.append(f"{key} BOOLEAN")
                elif isinstance(value, str) and len(value) > 255:
                    columns.append(f"{key} NCLOB")
                else:
                    python_type = type(value)
                    hana_type = type_mapping.get(python_type, "NVARCHAR(1000)")
                    columns.append(f"{key} {hana_type}")
            
            # Add standard audit columns
            if not any(col.startswith("created_at") for col in columns):
                columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            if not any(col.startswith("updated_at") for col in columns):
                columns.append("updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            if not any(col.startswith("deleted_at") for col in columns):
                columns.append("deleted_at TIMESTAMP NULL")
            
            # Create table
            create_sql = f"""
                CREATE TABLE {table_name} (
                    {', '.join(columns)}
                )
            """
            
            self.hana_client.execute_update(create_sql)
            logger.info(f"✅ Created HANA table: {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to create HANA table {table_name}: {e}")
            # Don't raise - allow operation to continue without table creation
    
    async def _update_status(self, task_id: str, state: TaskState, text: str = None, error: Dict = None):
        """Update task status"""
        status = TaskStatus(
            state=state,
            message=A2AMessage(
                role=MessageRole.SYSTEM,
                taskId=task_id,
                parts=[{
                    "kind": "text",
                    "text": text or f"Status: {state}"
                }]
            ) if text else None,
            error=error
        )
        
        self.status_updates[task_id].append(status)
        self.tasks[task_id]["status"] = state
        
        # Persist status update
        asyncio.create_task(self._persist_state())
    
    async def get_task_status(self, task_id: str) -> List[TaskStatus]:
        """Get task status history"""
        return self.status_updates.get(task_id, [])
    
    async def get_task_artifacts(self, task_id: str) -> List[TaskArtifact]:
        """Get task artifacts"""
        task = self.tasks.get(task_id, {})
        return task.get("artifacts", [])
    
    async def _register_with_ord(self, operation: str, result: Dict[str, Any], context_id: str):
        """Register data operations in ORD for lineage tracking"""
        # Register what was done but not the actual data
        # ORD tracks the metadata and lineage
        pass
    
    async def get_task_tracker_status(self, task_id: str = None) -> Dict[str, Any]:
        """Get comprehensive task tracking status"""
        if task_id:
            return self.task_tracker.get_task_status(task_id)
        else:
            return self.task_tracker.get_all_tasks_summary()
    
    async def get_active_help_requests(self) -> List[Dict[str, Any]]:
        """Get all active help requests"""
        return self.task_tracker.get_active_help_requests()
    
    async def get_advisor_stats(self) -> Dict[str, Any]:
        """Get AI advisor statistics"""
        return self.ai_advisor.get_advisor_stats()
    
    async def _load_persisted_state(self):
        """Load persisted data manager state from disk"""
        try:
            # Load tasks
            tasks_file = os.path.join(self._storage_path, "tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_id, task_info in tasks_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'created' in task_info:
                            task_info['created'] = datetime.fromisoformat(task_info['created'])
                        self.tasks[task_id] = task_info
                logger.info(f"✅ Loaded {len(self.tasks)} tasks from storage")
            
            # Load status updates
            status_file = os.path.join(self._storage_path, "status_updates.json")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    for task_id, status_list in status_data.items():
                        # Reconstruct TaskStatus objects
                        reconstructed_statuses = []
                        for status_dict in status_list:
                            reconstructed_statuses.append(TaskStatus(**status_dict))
                        self.status_updates[task_id] = reconstructed_statuses
                logger.info(f"✅ Loaded status updates for {len(self.status_updates)} tasks from storage")
            
            # Load operation history
            history_file = os.path.join(self._storage_path, "operation_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.operation_history = json.load(f)
                logger.info(f"✅ Loaded {len(self.operation_history)} operation records from storage")
                
        except Exception as e:
            logger.error(f"❌ Failed to load persisted state: {e}")
    
    async def _persist_state(self):
        """Persist data manager state to disk"""
        try:
            # Save tasks (convert datetime objects to strings)
            tasks_data = {}
            for task_id, task_info in self.tasks.items():
                task_data = task_info.copy()
                if 'created' in task_data:
                    task_data['created'] = task_data['created'].isoformat()
                tasks_data[task_id] = task_data
            
            tasks_file = os.path.join(self._storage_path, "tasks.json")
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Save status updates (convert TaskStatus objects to dicts)
            status_data = {}
            for task_id, status_list in self.status_updates.items():
                status_data[task_id] = [status.dict() if hasattr(status, 'dict') else status for status in status_list]
            
            status_file = os.path.join(self._storage_path, "status_updates.json")
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            # Save operation history for audit trail and acknowledgments
            history_file = os.path.join(self._storage_path, "operation_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.operation_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to persist data manager state: {e}")
    
    # ================================
    # A2A PROTOCOL COMPLIANCE METHODS
    # ================================
    
    async def _register_data_in_ord(
        self, 
        data: Any, 
        operation_type: str, 
        storage_location: StorageLocation, 
        context: Dict[str, Any] = None
    ) -> str:
        """Register data in ORD registry and return reference ID"""
        try:
            from app.ord_registry.service import ORDRegistryService
            from app.ord_registry.models import ORDDocument, DublinCoreMetadata, RegistrationRequest
            
            # Create Dublin Core metadata from context and data
            dublin_core = DublinCoreMetadata(
                title=f"Data from {operation_type} operation",
                description=f"Data created/processed by Data Manager Agent via {operation_type}",
                creator=["Data Manager Agent"],  # List as required by schema
                format="application/json",  # Default format
                type="Dataset",
                subject=[context.get("subject", "Data Management")],  # List as required by schema
                publisher="FinSight CIB",
                date=datetime.utcnow().isoformat(),
                identifier=str(uuid4())
            )
            
            # Create ORD document
            ord_doc = ORDDocument(
                namespace="finsight-cib.data-manager",
                localId=f"data-{operation_type}-{int(datetime.utcnow().timestamp())}",
                version="1.0.0",
                title=f"Data Manager {operation_type.title()} Result",
                shortDescription=f"Data resulting from {operation_type} operation",
                description=f"Data created/processed by Data Manager Agent through {operation_type} operation",
                packageLinks=[],
                links=[],
                entryPoints=[{
                    "type": "data-access",
                    "url": f"data-manager://storage/{storage_location.storage_type}",
                    "description": f"Access data via {storage_location.storage_type} storage"
                }],
                extensionInfo={
                    "storage_location": storage_location.model_dump(),
                    "operation_context": context or {},
                    "data_sample": str(data)[:500] if data else None  # First 500 chars as sample
                },
                dublinCore=dublin_core
            )
            
            # Register with ORD service
            ord_service = ORDRegistryService(base_url=self.ord_registry_url)
            registration_request = RegistrationRequest(
                ord_document=ord_doc,
                registrant_id="data_manager_agent"
            )
            
            registration_response = await ord_service.register_ord_document(
                ord_doc, 
                "data_manager_agent"
            )
            
            if registration_response and registration_response.registration_id:
                logger.info(f"✅ Data registered in ORD: {registration_response.registration_id}")
                return registration_response.registration_id
            else:
                raise Exception("ORD registration failed - no registration ID returned")
                
        except Exception as e:
            logger.error(f"❌ Failed to register data in ORD: {e}")
            # Return a fallback reference that indicates the error
            return f"ord-error-{int(datetime.utcnow().timestamp())}"
    
    async def _get_data_by_ord_reference(self, ord_reference: str, query_context: Dict[str, Any] = None) -> Any:
        """Retrieve data using ORD reference ID"""
        try:
            from app.ord_registry.service import ORDRegistryService
            
            # Get ORD document from registry
            ord_service = ORDRegistryService(base_url=self.ord_registry_url)
            registration = await ord_service.get_registration(ord_reference)
            
            if not registration:
                raise Exception(f"ORD reference {ord_reference} not found")
            
            # Extract storage location from ORD document extension info
            extension_info = registration.ord_document.extensionInfo or {}
            storage_location_data = extension_info.get("storage_location")
            
            if not storage_location_data:
                raise Exception(f"No storage location found in ORD reference {ord_reference}")
            
            storage_location = StorageLocation(**storage_location_data)
            
            # Retrieve data from storage location
            if storage_location.storage_type == "file" and storage_location.path:
                # Read from file
                file_path = self._resolve_path(storage_location.path)
                if storage_location.path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        return json.load(f)
                elif storage_location.path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    return df.to_dict('records')
                else:
                    with open(file_path, 'r') as f:
                        return f.read()
                        
            elif storage_location.storage_type in ["hana", "supabase"]:
                # Read from database
                data_request = DataRequest(
                    operation=DataOperation.read,
                    storage_type=StorageType(storage_location.storage_type),
                    query=query_context or {}
                )
                # Use database-specific query if available
                if storage_location.table:
                    data_request.query["table"] = storage_location.table
                if storage_location.schema:
                    data_request.query["schema"] = storage_location.schema
                    
                db_result = await self._read_from_database_v2(data_request)
                if hasattr(db_result, 'data'):
                    return db_result.data
                else:
                    raise Exception("No data returned from database query")
            else:
                raise Exception(f"Unsupported storage type: {storage_location.storage_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve data from ORD reference {ord_reference}: {e}")
            raise
    
    async def _resolve_ord_reference(self, ord_reference: str) -> Dict[str, Any]:
        """Resolve ORD reference to get metadata and access information"""
        try:
            from app.ord_registry.service import ORDRegistryService
            
            ord_service = ORDRegistryService(base_url=self.ord_registry_url)
            registration = await ord_service.get_registration(ord_reference)
            
            if not registration:
                raise Exception(f"ORD reference {ord_reference} not found")
            
            return {
                "ord_reference": ord_reference,
                "title": registration.ord_document.title,
                "description": registration.ord_document.description,
                "storage_location": registration.ord_document.extensionInfo.get("storage_location"),
                "access_methods": [ep.model_dump() for ep in registration.ord_document.entryPoints],
                "metadata": registration.ord_document.dublinCore.model_dump() if registration.ord_document.dublinCore else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve ORD reference {ord_reference}: {e}")
            raise
    
    async def _ensure_hana_table(self, table_name: str, sample_data: Dict[str, Any], schema: str = None):
        """Ensure HANA table exists with appropriate schema based on sample data"""
        try:
            # Check if table exists
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            try:
                # Try to get table info to see if it exists
                self.hana_client.get_table_info(table_name, schema)
                logger.info(f"Table {full_table_name} already exists")
                return
            except Exception:
                # Table doesn't exist, create it
                pass
            
            # Create table based on sample data structure
            columns = []
            for key, value in sample_data.items():
                if isinstance(value, str):
                    columns.append(f"{key} NVARCHAR(1000)")
                elif isinstance(value, int):
                    columns.append(f"{key} INTEGER")
                elif isinstance(value, float):
                    columns.append(f"{key} DECIMAL(15,4)")
                elif isinstance(value, bool):
                    columns.append(f"{key} BOOLEAN")
                elif isinstance(value, (dict, list)):
                    # Store complex types as JSON text
                    columns.append(f"{key} NCLOB")
                else:
                    # Default to NVARCHAR for unknown types
                    columns.append(f"{key} NVARCHAR(1000)")
            
            # Add metadata columns
            columns.extend([
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "created_by NVARCHAR(100) DEFAULT 'data_manager_agent'",
                "data_id NVARCHAR(36) DEFAULT SYSUUID"
            ])
            
            create_sql = f"CREATE TABLE {full_table_name} ({', '.join(columns)})"
            
            self.hana_client.execute_query(create_sql)
            logger.info(f"✅ Created HANA table: {full_table_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure HANA table {table_name}: {e}")
            # Don't raise - let the insert operation handle the error
    
    async def _create_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Create data in database with proper error handling"""
        try:
            # Use actual_data if available (from ORD resolution), otherwise use request.data
            actual_data = getattr(request, '_resolved_data', request.data)
            
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                
                # Convert data to format suitable for HANA insertion
                if isinstance(actual_data, list):
                    # Bulk insert
                    insert_data = actual_data
                else:
                    # Single record
                    insert_data = [actual_data]
                
                # Create table if it doesn't exist and we have data structure
                if insert_data and isinstance(insert_data[0], dict):
                    await self._ensure_hana_table(table_name, insert_data[0], schema)
                
                # Execute insert using HANA client
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                # Build INSERT statement
                if insert_data and isinstance(insert_data[0], dict):
                    columns = list(insert_data[0].keys())
                    placeholders = ', '.join(['?' for _ in columns])
                    insert_sql = f"INSERT INTO {full_table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Prepare data for batch insert
                    batch_data = [[record.get(col) for col in columns] for record in insert_data]
                    
                    # Execute batch insert
                    result = self.hana_client.execute_batch(insert_sql, batch_data)
                    rows_affected = len(batch_data)
                else:
                    # Handle non-dict data
                    insert_sql = f"INSERT INTO {full_table_name} (data_content) VALUES (?)"
                    self.hana_client.execute_query(insert_sql, [str(actual_data)])
                    rows_affected = 1
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=rows_affected
                    )
                )
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                
                # Ensure table exists
                if not self.supabase_client.validate_table_exists(table_name):
                    # Create the table if it doesn't exist
                    self.supabase_client.create_agent_data_table()
                
                # Insert data using Supabase client
                if isinstance(actual_data, list):
                    # Bulk insert
                    insert_response = self.supabase_client.insert(table_name, actual_data)
                else:
                    # Single record
                    insert_response = self.supabase_client.insert(table_name, [actual_data])
                
                if insert_response.error:
                    raise Exception(f"Supabase insert failed: {insert_response.error}")
                
                rows_affected = len(actual_data) if isinstance(actual_data, list) else 1
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=rows_affected
                    )
                )
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database create operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _read_from_database_v2(self, request: DataRequest) -> OperationResult:
        """Read data from database with proper error handling"""
        try:
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                where_clause = request.query.get("where") if request.query else None
                limit = request.query.get("limit", 1000) if request.query else 1000
                
                # Build SELECT statement
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                select_sql = f"SELECT * FROM {full_table_name}"
                params = []
                
                if where_clause:
                    select_sql += f" WHERE {where_clause}"
                    # Extract parameters if provided
                    params = request.query.get("params", []) if request.query else []
                
                select_sql += f" LIMIT {limit}"
                
                # Execute query
                query_result = self.hana_client.execute_query(select_sql, params)
                data = query_result.data
                
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=len(data)
                    )
                )
                result.data = data
                return result
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                columns = request.query.get("columns", "*") if request.query else "*"
                where_conditions = request.query.get("where") if request.query else None
                limit = request.query.get("limit", 1000) if request.query else 1000
                
                # Build Supabase query
                query_builder = self.supabase_client.select(
                    table=table_name,
                    columns=columns,
                    where=where_conditions,
                    limit=limit
                )
                
                if query_builder.error:
                    raise Exception(f"Supabase query failed: {query_builder.error}")
                
                data = query_builder.data or []
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=len(data)
                    )
                )
                result.data = data
                return result
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database read operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _update_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Update data in database with proper error handling"""
        try:
            # Use actual_data if available (from ORD resolution), otherwise use request.data
            actual_data = getattr(request, '_resolved_data', request.data)
            
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                where_clause = request.query.get("where") if request.query else None
                
                if not where_clause:
                    raise Exception("WHERE clause is required for update operations for safety")
                
                # Build UPDATE statement
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                if isinstance(actual_data, dict):
                    # Build SET clause from data
                    set_clauses = [f"{key} = ?" for key in actual_data.keys()]
                    set_clause = ", ".join(set_clauses)
                    
                    update_sql = f"UPDATE {full_table_name} SET {set_clause} WHERE {where_clause}"
                    params = list(actual_data.values())
                    
                    # Add WHERE parameters if provided
                    where_params = request.query.get("params", []) if request.query else []
                    params.extend(where_params)
                    
                    result = self.hana_client.execute_query(update_sql, params)
                    rows_affected = result.row_count if hasattr(result, 'row_count') else 0
                else:
                    # Handle non-dict data
                    update_sql = f"UPDATE {full_table_name} SET data_content = ? WHERE {where_clause}"
                    params = [str(actual_data)]
                    where_params = request.query.get("params", []) if request.query else []
                    params.extend(where_params)
                    
                    result = self.hana_client.execute_query(update_sql, params)
                    rows_affected = result.row_count if hasattr(result, 'row_count') else 0
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=rows_affected
                    )
                )
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                where_conditions = request.query.get("where") if request.query else None
                
                if not where_conditions:
                    raise Exception("WHERE conditions are required for update operations for safety")
                
                # Execute update using Supabase client
                update_data = actual_data if isinstance(actual_data, dict) else {"data_content": str(actual_data)}
                
                update_response = self.supabase_client.update(
                    table=table_name,
                    data=update_data,
                    where=where_conditions
                )
                
                if update_response.error:
                    raise Exception(f"Supabase update failed: {update_response.error}")
                
                rows_affected = update_response.count or 0
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=rows_affected
                    )
                )
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database update operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _delete_from_database_v2(self, request: DataRequest) -> OperationResult:
        """Delete data from database with proper error handling"""
        try:
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                where_clause = request.query.get("where") if request.query else None
                
                if not where_clause:
                    raise Exception("WHERE clause is required for delete operations for safety")
                
                # Build DELETE statement
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                delete_sql = f"DELETE FROM {full_table_name} WHERE {where_clause}"
                params = request.query.get("params", []) if request.query else []
                
                # Execute delete
                result = self.hana_client.execute_query(delete_sql, params)
                rows_affected = result.row_count if hasattr(result, 'row_count') else 0
                
                return OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=rows_affected
                    )
                )
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                where_conditions = request.query.get("where") if request.query else None
                
                if not where_conditions:
                    raise Exception("WHERE conditions are required for delete operations for safety")
                
                # Execute delete using Supabase client
                delete_response = self.supabase_client.delete(
                    table=table_name,
                    where=where_conditions
                )
                
                if delete_response.error:
                    raise Exception(f"Supabase delete failed: {delete_response.error}")
                
                rows_affected = delete_response.count or 0
                
                return OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=rows_affected
                    )
                )
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database delete operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _list_in_database_v2(self, request: DataRequest) -> OperationResult:
        """List data in database with proper error handling"""
        try:
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                limit = request.query.get("limit", 100) if request.query else 100
                offset = request.query.get("offset", 0) if request.query else 0
                
                # Build SELECT with pagination
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                list_sql = f"SELECT * FROM {full_table_name} LIMIT {limit} OFFSET {offset}"
                
                query_result = self.hana_client.execute_query(list_sql)
                data = query_result.data
                
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=len(data)
                    )
                )
                result.data = data
                return result
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                limit = request.query.get("limit", 100) if request.query else 100
                offset = request.query.get("offset", 0) if request.query else 0
                
                # Use Supabase client to list with pagination
                list_response = self.supabase_client.select(
                    table=table_name,
                    columns="*",
                    limit=limit,
                    offset=offset
                )
                
                if list_response.error:
                    raise Exception(f"Supabase list failed: {list_response.error}")
                
                data = list_response.data or []
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=len(data)
                    )
                )
                result.data = data
                return result
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database list operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    async def _exists_in_database_v2(self, request: DataRequest) -> OperationResult:
        """Check if data exists in database with proper error handling"""
        try:
            if request.storage_type == StorageType.HANA and self.hana_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                schema = request.query.get("schema") if request.query else None
                where_clause = request.query.get("where") if request.query else None
                
                if not where_clause:
                    raise Exception("WHERE clause is required for exists operations")
                
                # Build SELECT COUNT query
                if schema:
                    full_table_name = f"{schema}.{table_name}"
                else:
                    full_table_name = table_name
                
                exists_sql = f"SELECT COUNT(*) as count FROM {full_table_name} WHERE {where_clause}"
                params = request.query.get("params", []) if request.query else []
                
                query_result = self.hana_client.execute_query(exists_sql, params)
                count = query_result.data[0]["count"] if query_result.data else 0
                exists = count > 0
                
                result = OperationResult(
                    storage_type="hana",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="hana",
                        database="HANA",
                        schema=schema,
                        table=table_name,
                        row_count=count
                    )
                )
                result.data = {"exists": exists, "count": count}
                return result
                
            elif request.storage_type == StorageType.SUPABASE and self.supabase_client:
                table_name = request.query.get("table", "a2a_data") if request.query else "a2a_data"
                where_conditions = request.query.get("where") if request.query else None
                
                if not where_conditions:
                    raise Exception("WHERE conditions are required for exists operations")
                
                # Use Supabase client to check existence
                exists_response = self.supabase_client.select(
                    table=table_name,
                    columns="count",  # Count only
                    where=where_conditions,
                    limit=1
                )
                
                if exists_response.error:
                    raise Exception(f"Supabase exists check failed: {exists_response.error}")
                
                count = exists_response.count or 0
                exists = count > 0
                
                result = OperationResult(
                    storage_type="supabase",
                    status=OperationStatus.SUCCESS,
                    location=StorageLocation(
                        storage_type="supabase",
                        database="supabase",
                        table=table_name,
                        row_count=count
                    )
                )
                result.data = {"exists": exists, "count": count}
                return result
            else:
                raise Exception(f"No suitable database client for {request.storage_type}")
                
        except Exception as e:
            logger.error(f"Database exists operation failed: {e}")
            return OperationResult(
                storage_type=str(request.storage_type),
                status=OperationStatus.FAILED,
                error=str(e)
            )


# Router implementation
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

def create_data_manager_router(agent: DataManagerAgent) -> APIRouter:
    router = APIRouter(prefix="/a2a/data_manager/v1", tags=["Data Manager Agent"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        return agent.agent_card.model_dump()
    
    @router.post("/messages")
    async def process_message(message: A2AMessage):
        response = await agent.process_message(message)
        return response.model_dump()
    
    @router.get("/tasks/{task_id}/status")
    async def get_task_status(task_id: str):
        status_list = await agent.get_task_status(task_id)
        return {"status": [s.model_dump() for s in status_list]}
    
    @router.get("/tasks/{task_id}/artifacts")
    async def get_task_artifacts(task_id: str):
        artifacts = await agent.get_task_artifacts(task_id)
        return {"artifacts": [a.model_dump() for a in artifacts]}
    
    @router.get("/queue/status")
    async def get_queue_status():
        """Get message queue status for Data Manager"""
        if agent and agent.message_queue:
            return agent.message_queue.get_queue_status()
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message queue not available"}
            )

    @router.get("/queue/messages/{message_id}")
    async def get_message_status(message_id: str):
        """Get status of a specific message"""
        if agent and agent.message_queue:
            status = agent.message_queue.get_message_status(message_id)
            if status:
                return status
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Message not found"}
                )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message queue not available"}
            )

    @router.delete("/queue/messages/{message_id}")
    async def cancel_message(message_id: str):
        """Cancel a queued or processing message"""
        if agent and agent.message_queue:
            cancelled = await agent.message_queue.cancel_message(message_id)
            if cancelled:
                return {"message": "Message cancelled successfully"}
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Message not found or cannot be cancelled"}
                )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message queue not available"}
            )

    @router.get("/health")
    async def health_check():
        """Health check endpoint for Data Manager"""
        queue_info = {}
        if agent and agent.message_queue:
            queue_status = agent.message_queue.get_queue_status()
            queue_info = {
                "queue_depth": queue_status["queue_status"]["queue_depth"],
                "processing_count": queue_status["queue_status"]["processing_count"],
                "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
                "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
            }
        
        # Get cache statistics
        cache_info = {}
        if agent and hasattr(agent, 'cache_manager'):
            try:
                cache_info = await agent.cache_manager.get_stats()
            except Exception as e:
                cache_info = {"error": str(e)}
        
        return {
            "status": "healthy",
            "agent": "Data Manager Agent (BDC Core)",
            "version": "2.0.0", 
            "protocol_version": "0.2.9",
            "timestamp": datetime.utcnow().isoformat(),
            "message_queue": queue_info,
            "cache_stats": cache_info
        }
    
    @router.get("/cache/stats")
    async def get_cache_stats():
        """Get detailed cache statistics"""
        if not agent or not hasattr(agent, 'cache_manager'):
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        try:
            return await agent.cache_manager.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cache stats error: {str(e)}")
    
    @router.post("/cache/invalidate")
    async def invalidate_cache(namespace: str, key: str = None):
        """Invalidate cache entries"""
        if not agent or not hasattr(agent, 'cache_manager'):
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        try:
            await agent.cache_manager.invalidate(namespace, key)
            return {"status": "success", "message": f"Invalidated {namespace}:{key or 'all'}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cache invalidation error: {str(e)}")
    
    # Direct data endpoints for non-A2A access
    @router.post("/data/crud")
    async def crud_operation(request: DataRequest):
        """Direct CRUD endpoint"""
        agent_message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(kind="data", data=request.model_dump())
            ]
        )
        
        response = await agent.process_message(agent_message)
        
        # Wait for completion (simplified for direct access)
        task_id = response.taskId
        max_wait = 30  # seconds
        
        for _ in range(max_wait):
            status_list = await agent.get_task_status(task_id)
            if status_list:
                latest = status_list[-1]
                if latest.state in [TaskState.COMPLETED, TaskState.FAILED]:
                    if latest.state == TaskState.COMPLETED:
                        artifacts = await agent.get_task_artifacts(task_id)
                        if artifacts:
                            return artifacts[0].parts[0].data
                    else:
                        return JSONResponse(
                            status_code=500,
                            content={"error": latest.error}
                        )
            await asyncio.sleep(1)
        
        return JSONResponse(
            status_code=408,
            content={"error": "Operation timeout"}
        )
    
    return router
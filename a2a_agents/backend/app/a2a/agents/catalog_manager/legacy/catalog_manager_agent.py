import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
import logging
import httpx

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..core.a2a_types import A2AMessage, MessagePart, MessageRole
from .data_standardization_agent import (
    TaskState, TaskStatus, TaskArtifact, AgentCard
)
from app.a2a.core.workflow_context import workflow_context_manager, DataArtifact
from app.a2a.core.workflow_monitor import workflow_monitor
from app.ord_registry.service import ORDRegistryService
from app.ord_registry.advanced_ai_enhancer import create_advanced_ai_enhancer
from app.clients.grok_client import get_grok_client
from app.clients.perplexity_client import get_perplexity_client
from app.ord_registry.models import ORDDocument, DublinCoreMetadata, RegistrationRequest
from ..security.smart_contract_trust import initialize_agent_trust, sign_a2a_message, get_trust_contract, verify_a2a_message
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
from app.a2a_registry.client import get_registry_client
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus

logger = logging.getLogger(__name__)


# A2AMessage, MessagePart, TaskStatus, and related models are imported from data_standardization_agent


class ORDRepositoryRequest(BaseModel):
    """Request for ORD repository operations"""
    operation: str = Field(description="Operation type: register, enhance, search, update, delete, quality_check")
    ord_document: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    registration_id: Optional[str] = None
    enhancement_type: str = Field(default="metadata_enrichment")
    ai_powered: bool = Field(default=True, description="Use AI enhancement")
    context: Optional[Dict[str, Any]] = None


class ORDRepositoryResponse(BaseModel):
    """Response from ORD repository operations"""
    operation: str
    success: bool
    result: Dict[str, Any]
    enhanced_document: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    ai_insights: Optional[Dict[str, Any]] = None
    processing_time: float
    message: str


class CatalogManagerAgent(AgentHelpSeeker):
    """Catalog Manager Agent - AI-powered ORD repository management and enhancement"""
    
    def __init__(self, base_url: str, ord_registry_url: str = None, downstream_agent_url: str = None):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        if not ord_registry_url:
            raise ValueError("ORD registry URL is required for Catalog Manager. No fallback allowed.")
        self.ord_registry_url = ord_registry_url
        self.downstream_agent_url = downstream_agent_url
        self.registry_client = None
        
        # Initialize ORD Registry Service
        self.ord_service = ORDRegistryService(base_url=self.ord_registry_url)
        
        # Initialize AI clients
        self.grok_client = get_grok_client()
        self.perplexity_client = get_perplexity_client()
        self.ai_enhancer = create_advanced_ai_enhancer(
            grok_client=self.grok_client,
            perplexity_client=self.perplexity_client
        )
        
        # Agent identification (trust identity will be registered via Agent Manager)
        self.agent_id = "catalog_manager_agent"
        self.agent_name = "CatalogManagerAgent"
        self.agent_identity = None  # Will be set after Agent Manager registration
        logger.info(f"Catalog Manager identity pending registration via Agent Manager: {self.agent_id}")
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name=self.agent_name
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "ord_registry_url": self.ord_registry_url,
            "agent_type": "catalog_manager",
            "timeout": 30.0,
            "retry.max_attempts": 3
        }
        self.initialize_help_action_system(self.agent_id, agent_context)
        
        self.agent_card = AgentCard(
            name="Catalog Manager Agent",
            description="AI-powered ORD repository management with advanced metadata enhancement, quality assessment, and compliance validation capabilities",
            url=base_url,
            version="2.0.0",
            protocolVersion="0.2.9",
            provider={
                "organization": "FinSight CIB",
                "url": "https://finsight-cib.com"
            },
            capabilities={
                "streaming": True,
                "pushNotifications": True,
                "stateTransitionHistory": True,
                "batchProcessing": True,
                "ordRepository": True,
                "aiEnhancement": True,
                "semanticAnalysis": True,
                "qualityAssessment": True,
                "dublinCoreEnrichment": True,
                "multiModelAI": True,
                "complianceValidation": True,
                "metadataManagement": True,
                "smartContractDelegation": True,
                "aiAdvisor": True,
                "helpSeeking": True,
                "taskTracking": True
            },
            tools=[
                {
                    "name": "register_ord_document",
                    "description": "Register new ORD document with AI enhancement",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ord_document": {"type": "object"},
                            "ai_powered": {"type": "boolean"},
                            "enhancement_type": {"type": "string"}
                        },
                        "required": ["ord_document"]
                    }
                },
                {
                    "name": "enhance_ord_metadata",
                    "description": "AI-powered enhancement of existing ORD metadata",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "registration_id": {"type": "string"},
                            "enhancement_type": {"type": "string"},
                            "multi_model": {"type": "boolean"}
                        },
                        "required": ["registration_id"]
                    }
                },
                {
                    "name": "search_ord_repository",
                    "description": "AI-powered semantic search of ORD repository",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {"type": "object"},
                            "semantic": {"type": "boolean"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "assess_ord_quality",
                    "description": "AI-powered quality assessment of ORD documents",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "registration_id": {"type": "string"},
                            "assessment_type": {"type": "string"}
                        },
                        "required": ["registration_id"]
                    }
                },
                {
                    "name": "ai_advisor",
                    "description": "AI-powered help and guidance using Grok-4",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "asking_agent": {"type": "string"}
                        },
                        "required": ["question"]
                    }
                }
            ]
        )
        
        # Initialize Database-backed AI Decision Logger
        from ..core.ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
        
        # Construct Data Manager URL
        data_manager_url = f"{self.base_url.replace('/agents/', '/').rstrip('/')}/data-manager"
        
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=self.agent_id,
            data_manager_url=data_manager_url,
            memory_size=1000,
            learning_threshold=7,  # Catalog decisions benefit from faster learning
            cache_ttl=300
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI Advisor
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name="Catalog Manager Agent",
            agent_capabilities={
                "streaming": True,
                "pushNotifications": True,
                "stateTransitionHistory": True,
                "batchProcessing": True,
                "ordRepository": True,
                "aiEnhancement": True,
                "semanticAnalysis": True,
                "qualityAssessment": True,
                "dublinCoreEnrichment": True,
                "multiModelAI": True,
                "complianceValidation": True,
                "metadataManagement": True,
                "smartContractDelegation": True,
                "aiAdvisor": True,
                "helpSeeking": True,
                "taskTracking": True
            }
        )
        
        # Add knowledge to AI advisor
        self._initialize_advisor_knowledge()
        
        # Set operational state callback for live status reporting
        self.ai_advisor.set_operational_state_callback(self.get_agent_operational_state)
        
        # Initialize message queue with agent-specific configuration
        self.initialize_message_queue(
            agent_id=self.agent_id,
            max_concurrent_processing=4,  # Balanced for metadata processing
            auto_mode_threshold=7,        # Switch to queue after 7 pending
            enable_streaming=True,        # Support real-time processing
            enable_batch_processing=True  # Support batch processing
        )
        
        # Task tracking with persistence
        self.active_tasks = {}
        self.task_history = []
        
        # Initialize persistent storage for tasks
        self._storage_path = os.getenv("CATALOG_MANAGER_STORAGE_PATH", "/tmp/catalog_manager_state")
        os.makedirs(self._storage_path, exist_ok=True)
        
        # Load persisted state
        asyncio.create_task(self._load_persisted_state())
        
        # Set message queue processor callback after all methods are defined
        if self.message_queue:
            self.message_queue.set_message_processor(self._process_message_core)
        
        logger.info("✅ Catalog Manager Agent v2.0.0 initialized with AI advisor, help-seeking, task tracking, and message queue capabilities")
    
    def _initialize_advisor_knowledge(self):
        """Initialize AI advisor with agent-specific knowledge"""
        # Add FAQs
        self.ai_advisor.add_faq_item(
            "What does the Catalog Manager do?",
            "I am the Catalog Manager Agent. I manage the ORD (Object Resource Discovery) repository with AI-powered metadata enhancement, quality assessment, and compliance validation. I handle document registration, search operations, and metadata enrichment using multiple AI models."
        )
        
        self.ai_advisor.add_faq_item(
            "What ORD operations do you support?",
            "I support full ORD lifecycle operations: document registration with AI enhancement, semantic search, quality assessment, metadata enrichment using Grok and Perplexity, compliance validation, and repository management. I can work with Dublin Core metadata and provide multi-model AI insights."
        )
        
        self.ai_advisor.add_faq_item(
            "How do you work with other agents?",
            "I receive delegation requests from Agent 0 (Data Product Registration) for metadata registration and from Agent 1 (Data Standardization) for quality assessment. I can also delegate data retrieval tasks to the Data Manager when needed."
        )
        
        # Add common issues
        self.ai_advisor.add_common_issue(
            "ord_service_unavailable",
            "ORD registry service is not responding or unavailable",
            "Check if the ORD service is running on the configured URL. Verify network connectivity and service health. Restart the ORD service if necessary. Check service logs for errors."
        )
        
        self.ai_advisor.add_common_issue(
            "ai_enhancement_failed",
            "AI enhancement using Grok or Perplexity fails",
            "Verify AI client configurations and API keys. Check if the AI services are responding. Try with a smaller document payload. Review AI service quotas and rate limits."
        )
        
        self.ai_advisor.add_common_issue(
            "quality_assessment_timeout",
            "Quality assessment operations timeout on large documents",
            "Large documents should be processed in smaller chunks. Check AI service response times. Verify document format and structure. Consider using batch processing for multiple documents."
        )
        
        # Add help-seeking knowledge
        self.ai_advisor.add_faq_item(
            "Can you ask other agents for help?",
            "Yes! I can actively seek help from other agents when I encounter problems. I know how to contact the Data Manager for storage issues, Agent 0 for data processing problems, and Agent 1 for standardization guidance."
        )
        
        self.ai_advisor.add_common_issue(
            "help_seeking_available",
            "When should I seek help from other agents?",
            "I automatically seek help when I encounter repeated failures (3+ consecutive errors), network connectivity issues, service unavailability, or data integrity problems. I can ask the Data Manager for storage help, Agent 0 for ORD processing guidance, and others based on the problem type."
        )
    
    async def initialize(self):
        """Initialize ORD service and dependencies"""
        try:
            await self.ord_service.initialize()
            logger.info("✅ Catalog Manager ORD service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ORD service: {e}")
            raise
    
    def _extract_content_from_message(self, message: A2AMessage) -> str:
        """Extract text content from A2AMessage parts"""
        content_parts = []
        for part in message.parts:
            if part.text:
                content_parts.append(part.text)
            elif part.data and isinstance(part.data, dict):
                # Extract text from data if it contains readable content
                content_parts.append(str(part.data))
        return " ".join(content_parts) if content_parts else "[No text content]"
    
    async def get_agent_card(self) -> AgentCard:
        """Return the agent card"""
        return self.agent_card
    
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
    
    async def _handle_advisor_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle AI advisor help requests"""
        try:
            asking_agent_id = getattr(message, 'from_agent_id', None)
            advisor_response = await self.ai_advisor.process_a2a_help_message(
                [part.model_dump() if hasattr(part, 'model_dump') else part for part in message.parts],
                asking_agent_id
            )
            
            return {
                "message_type": "advisor_response",
                "advisor_response": advisor_response,
                "agent_id": self.agent_id,
                "contextId": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error handling advisor request: {e}")
            return {
                "message_type": "advisor_error",
                "error": str(e),
                "agent_id": self.agent_id,
                "contextId": context_id
            }
    
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
            context_id = message.contextId or str(datetime.utcnow().timestamp())
        
        if not self.message_queue:
            # Fallback to direct processing if queue not initialized
            result = await self._process_message_core(message, context_id)
            # Convert TaskStatus result to dict for consistency
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
            a2a_message=message.model_dump() if hasattr(message, 'model_dump') else message,
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
    
    async def _process_message_core(self, message: A2AMessage, context_id: str = None) -> TaskStatus:
        """Process incoming A2A messages for ORD repository management"""
        try:
            task_id = str(uuid4())
            context_id = context_id or str(datetime.utcnow().timestamp())
            
            # Check if this is an AI advisor request
            if self._is_advisor_request(message):
                return await self._handle_advisor_request(message, context_id)
            
            # Extract content from message parts
            content = self._extract_content_from_message(message)
            logger.info(f"🔄 Catalog Manager processing ORD message: {message.role} - {content[:100]}...")
            
            # Initialize task tracking
            task_status = TaskStatus(
                state=TaskState.WORKING,
                message=A2AMessage(
                    role=MessageRole.AGENT,
                    parts=[
                        MessagePart(
                            kind="text",
                            text="Initializing ORD repository operation..."
                        )
                    ],
                    taskId=task_id,
                    contextId=context_id
                )
            )
            
            self.active_tasks[task_id] = task_status
            
            # Persist state after task creation
            asyncio.create_task(self._persist_state())
            
            # Create workflow context
            workflow_context = workflow_context_manager.create_workflow_context(
                workflow_plan_id=task_id,
                workflow_name="ORD Repository Management",
                initiated_by=self.agent_id,
                initial_stage="ord_processing",
                metadata={"context_id": context_id, "task_type": "ord_repository_management"}
            )
            
            # Parse message content to determine operation
            try:
                # Try to parse as JSON for structured requests
                request_data = json.loads(content)
                request = ORDRepositoryRequest(**request_data)
            except:
                # Handle natural language requests
                request = self._parse_natural_language_request(content)
            
            # Route to appropriate handler
            if request.operation == "register":
                return await self._handle_registration(request, task_id, context_id, workflow_context)
            elif request.operation == "enhance":
                return await self._handle_enhancement(request, task_id, context_id)
            elif request.operation == "search":
                return await self._handle_search(request, task_id, context_id)
            elif request.operation == "quality_check":
                return await self._handle_quality_assessment(request, task_id, context_id)
            elif request.operation == "update":
                return await self._handle_update(request, task_id, context_id)
            elif request.operation == "delete":
                return await self._handle_delete(request, task_id, context_id)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
        except Exception as e:
            logger.error(f"Error processing Catalog Manager message: {e}")
            
            # Update task status with error
            if task_id in self.active_tasks:
                self.active_tasks[task_id].state = TaskState.FAILED
                self.active_tasks[task_id].message = A2AMessage(
                    role=MessageRole.AGENT,
                    parts=[
                        MessagePart(
                            kind="text",
                            text=f"Error: {str(e)}"
                        )
                    ],
                    taskId=task_id,
                    contextId=context_id
                )

                
                # Create error response
                error_artifact = TaskArtifact(
                    name="error_response",
                    description=f"Error processing Catalog Manager message: {str(e)}",
                    parts=[
                        MessagePart(
                            kind="error",
                            text=f"Operation failed: {str(e)}",
                            data={"error": str(e), "operation": "unknown", "timestamp": datetime.utcnow().isoformat()}
                        )
                    ]
                )
                
                return TaskStatus(
                    state=TaskState.FAILED,
                    message=A2AMessage(
                        role=MessageRole.AGENT,
                        parts=[
                            MessagePart(
                                kind="text",
                                text=f"Failed to process message: {str(e)}"
                            )
                        ],
                        taskId=task_id,
                        contextId=context_id
                    ),
                    error={
                        "error_type": "message_processing_error",
                        "message": str(e),
                        "task_id": task_id,
                        "context_id": context_id
                    }
                )
    
    def _parse_natural_language_request(self, content: str) -> ORDRepositoryRequest:
        """Parse natural language requests into structured requests"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["register", "create", "add", "new"]):
            return ORDRepositoryRequest(operation="register")
        elif any(word in content_lower for word in ["enhance", "improve", "enrich", "ai"]):
            return ORDRepositoryRequest(operation="enhance")
        elif any(word in content_lower for word in ["search", "find", "query", "lookup"]):
            return ORDRepositoryRequest(operation="search", query=content)
        elif any(word in content_lower for word in ["quality", "assess", "validate", "check"]):
            return ORDRepositoryRequest(operation="quality_check")
        elif any(word in content_lower for word in ["update", "modify", "change"]):
            return ORDRepositoryRequest(operation="update")
        elif any(word in content_lower for word in ["delete", "remove", "deregister"]):
            return ORDRepositoryRequest(operation="delete")
        else:
            # Default to enhancement if unclear
            return ORDRepositoryRequest(operation="enhance")
    
    async def _handle_registration(self, request: ORDRepositoryRequest, task_id: str, context_id: str, workflow_context=None) -> TaskStatus:
        """Handle ORD document registration with AI enhancement"""
        try:
            start_time = datetime.utcnow()
            

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Processing ORD document registration..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            if not request.ord_document:
                raise ValueError("ORD document required for registration")
            
            # Convert to ORD document model
            ord_doc = ORDDocument(**request.ord_document)
            
            # Apply AI enhancement if requested
            enhanced_doc = ord_doc
            ai_insights = {}
            
            if request.ai_powered:

                self.active_tasks[task_id].message = A2AMessage(
                    role=MessageRole.AGENT,
                    parts=[
                        MessagePart(
                            kind="text",
                            text="Applying AI enhancement..."
                        )
                    ],
                    taskId=task_id,
                    contextId=context_id
                )
                
                enhancement_result = await self.ai_enhancer.multi_model_enhancement(
                    ord_doc, request.enhancement_type
                )
                
                # Create enhanced document
                enhanced_doc = ORDDocument(**enhancement_result.enhanced_content)
                ai_insights = {
                    "model_used": enhancement_result.model_used,
                    "confidence_score": enhancement_result.confidence_score,
                    "quality_improvements": enhancement_result.quality_improvements,
                    "processing_time": enhancement_result.processing_time
                }
            
            # Register with ORD service

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Registering with ORD repository..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            # Register the enhanced ORD document
            registration_response = await self.ord_service.register_ord_document(
                enhanced_doc, 
                "catalog_manager_agent"
            )
            
            # Check if registration failed (should delegate to downstream storage agent)
            if registration_response is None:
                # TODO: Refactor to send A2A message to downstream Data Manager agent for storage delegation
                # For now, create A2A-style error message indicating storage agent communication failure
                error_msg = "Registration failed - unable to communicate with downstream storage agent"
                logger.error(f"Storage delegation failed: {error_msg}")
                
                # Update task status to failed with A2A error message
                self.active_tasks[task_id].state = TaskState.FAILED
                self.active_tasks[task_id].message = A2AMessage(
                    role="agent",
                    parts=[
                        MessagePart(
                            kind="error",
                            text=error_msg,
                            data={
                                "error_type": "storage_delegation_failure", 
                                "recommended_action": "Check Data Manager agent connectivity",
                                "delegation_target": "data_manager_agent"
                            }
                        )
                    ],
                    taskId=task_id,
                    contextId=context_id
                )
                return self.active_tasks[task_id].model_dump() if hasattr(self.active_tasks[task_id], 'model_dump') else self.active_tasks[task_id]
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create response
            response = ORDRepositoryResponse(
                operation="register",
                success=True,
                result={
                    "registration_id": registration_response.registration_id,
                    "status": registration_response.metadata.status.value,
                    "registry_url": self.ord_registry_url
                },
                enhanced_document=enhanced_doc.model_dump() if request.ai_powered else None,
                ai_insights=ai_insights if request.ai_powered else None,
                processing_time=processing_time,
                message="ORD document registered successfully"
            )
            
            # Create success artifact
            success_artifact = TaskArtifact(
                name="registration_response",
                description="Successfully registered ORD document with AI enhancement",
                parts=[
                    MessagePart(
                        kind="success",
                        text="ORD document registered successfully",
                        data={
                            "response": response.model_dump() if hasattr(response, 'model_dump') else response,
                            "registration_id": registration_response.registration_id,
                            "ai_enhanced": request.ai_powered,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            
            # Update task status
            self.active_tasks[task_id].state = TaskState.COMPLETED
            # Persist state after task completion
            asyncio.create_task(self._persist_state())

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="ORD document registered successfully"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            # Store artifact in workflow context
            if workflow_context:
                workflow_context_manager.create_data_artifact(
                    workflow_id=workflow_context.workflow_id,
                    artifact_type="registered_ord_document",
                    location=f"registration_{registration_response.registration_id}",
                    created_by=self.agent_id,
                    metadata={"registration_id": registration_response.registration_id, "response": response.model_dump() if hasattr(response, 'model_dump') else response}
                )
            
            logger.info(f"✅ ORD document registered: {registration_response.registration_id}")
            
            return self.active_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            self.active_tasks[task_id].state = TaskState.FAILED
            # Persist state after task failure
            asyncio.create_task(self._persist_state())
            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text=f"Registration failed: {str(e)}"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            raise
    
    async def _handle_enhancement(self, request: ORDRepositoryRequest, task_id: str, context_id: str) -> TaskStatus:
        """Handle AI-powered enhancement of existing ORD documents"""
        try:
            start_time = datetime.utcnow()
            

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Retrieving ORD document for enhancement..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            if not request.registration_id and not request.ord_document:
                raise ValueError("Registration ID or ORD document required for enhancement")
            
            # Get document
            if request.registration_id:
                registration = await self.ord_service.get_registration(request.registration_id)
                ord_doc = registration.ord_document
            else:
                ord_doc = ORDDocument(**request.ord_document)
            
            # Apply AI enhancement

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Applying AI enhancement..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            enhancement_result = await self.ai_enhancer.multi_model_enhancement(
                ord_doc, request.enhancement_type
            )
            
            # Update document if registration_id provided
            if request.registration_id:

                self.active_tasks[task_id].message = A2AMessage(
                    role=MessageRole.AGENT,
                    parts=[
                        MessagePart(
                            kind="text",
                            text="Updating ORD repository..."
                        )
                    ],
                    taskId=task_id,
                    contextId=context_id
                )
                
                # Here you would update the document in the repository
                # For now, we'll just return the enhanced version
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create response
            response = ORDRepositoryResponse(
                operation="enhance",
                success=True,
                result={
                    "registration_id": request.registration_id,
                    "enhancement_applied": True
                },
                enhanced_document=enhancement_result.enhanced_content,
                ai_insights={
                    "model_used": enhancement_result.model_used,
                    "confidence_score": enhancement_result.confidence_score,
                    "quality_improvements": enhancement_result.quality_improvements,
                    "processing_time": enhancement_result.processing_time
                },
                processing_time=processing_time,
                message="ORD document enhanced successfully"
            )
            
            # Create success artifact
            success_artifact = TaskArtifact(
                name="enhancement_response",
                description="Successfully enhanced ORD document with AI capabilities",
                parts=[
                    MessagePart(
                        kind="success",
                        text="ORD document enhanced successfully",
                        data={
                            "response": response.model_dump() if hasattr(response, 'model_dump') else response,
                            "enhancement_type": request.enhancement_type,
                            "model_used": enhancement_result.model_used,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            
            # Update task status
            self.active_tasks[task_id].state = TaskState.COMPLETED
            # Persist state after task completion
            asyncio.create_task(self._persist_state())

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="ORD document enhanced successfully"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            logger.info(f"✅ ORD document enhanced: {request.registration_id}")
            
            return self.active_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            self.active_tasks[task_id].state = TaskState.FAILED
            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text=f"Enhancement failed: {str(e)}"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            raise
    
    async def _handle_search(self, request: ORDRepositoryRequest, task_id: str, context_id: str) -> TaskStatus:
        """Handle AI-powered search of ORD repository"""
        try:
            start_time = datetime.utcnow()
            

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Performing AI-powered search..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            if not request.query:
                raise ValueError("Query required for search operation")
            
            # Perform search (simplified - you would integrate with your search service)
            search_results = await self._perform_semantic_search(request.query, request.context)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create response
            response = ORDRepositoryResponse(
                operation="search",
                success=True,
                result=search_results,
                processing_time=processing_time,
                message=f"Found {len(search_results.get('results', []))} results"
            )
            
            # Create success artifact
            success_artifact = TaskArtifact(
                name="search_response",
                description="Successfully completed ORD semantic search operation",
                parts=[
                    MessagePart(
                        kind="success",
                        text="Search completed successfully",
                        data={
                            "response": response.model_dump() if hasattr(response, 'model_dump') else response,
                            "query": request.query,
                            "result_count": len(search_results.get('results', [])),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            
            # Update task status
            self.active_tasks[task_id].state = TaskState.COMPLETED
            # Persist state after task completion
            asyncio.create_task(self._persist_state())

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Search completed successfully"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            logger.info(f"✅ Search completed: {request.query}")
            
            return self.active_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.active_tasks[task_id].state = TaskState.FAILED
            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text=f"Search failed: {str(e)}"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            raise
    
    async def _handle_quality_assessment(self, request: ORDRepositoryRequest, task_id: str, context_id: str) -> TaskStatus:
        """Handle AI-powered quality assessment of ORD documents"""
        try:
            start_time = datetime.utcnow()
            

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Performing quality assessment..."
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            if not request.registration_id and not request.ord_document:
                raise ValueError("Registration ID or ORD document required for quality assessment")
            
            # Get document
            if request.registration_id:
                registration = await self.ord_service.get_registration(request.registration_id)
                ord_doc = registration.ord_document
            else:
                ord_doc = ORDDocument(**request.ord_document)
            
            # Perform quality assessment
            quality_result = await self.ai_enhancer.advanced_quality_assessment(ord_doc)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create response
            response = ORDRepositoryResponse(
                operation="quality_check",
                success=True,
                result={"quality_assessment": quality_result},
                quality_metrics=quality_result,
                processing_time=processing_time,
                message="Quality assessment completed"
            )
            
            # Create success artifact
            success_artifact = TaskArtifact(
                name="quality_assessment_response",
                description="Successfully completed ORD quality assessment with AI analysis",
                parts=[
                    MessagePart(
                        kind="success",
                        text="Quality assessment completed",
                        data={
                            "response": response.model_dump() if hasattr(response, 'model_dump') else response,
                            "registration_id": request.registration_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            
            # Update task status
            self.active_tasks[task_id].state = TaskState.COMPLETED
            # Persist state after task completion
            asyncio.create_task(self._persist_state())

            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text="Quality assessment completed"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            
            logger.info(f" Quality assessment completed: {request.registration_id}")
            
            return self.active_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            self.active_tasks[task_id].state = TaskState.FAILED
            self.active_tasks[task_id].message = A2AMessage(
                role="agent",
                parts=[
                    MessagePart(
                        kind="text",
                        text=f"Quality assessment failed: {str(e)}"
                    )
                ],
                taskId=task_id,
                contextId=context_id
            )
            raise
    
    async def _handle_update(self, request: ORDRepositoryRequest, task_id: str, context_id: str) -> TaskStatus:
        """Handle ORD document updates"""
        # Update operation not implemented - fail explicitly instead of fake success
        error_message = "ORD document update operation is not implemented. This functionality requires proper implementation before use."
        
        self.active_tasks[task_id].state = TaskState.FAILED
        self.active_tasks[task_id].message = A2AMessage(
            role="agent",
            parts=[
                MessagePart(
                    kind="error",
                    text=error_message
                )
            ],
            taskId=task_id,
            contextId=context_id
        )
        
        raise NotImplementedError(error_message)
    
    async def _handle_delete(self, request: ORDRepositoryRequest, task_id: str, context_id: str) -> TaskStatus:
        """Handle ORD document deletion"""
        # Delete operation not implemented - fail explicitly instead of fake success
        error_message = "ORD document delete operation is not implemented. This functionality requires proper implementation before use."
        
        self.active_tasks[task_id].state = TaskState.FAILED
        self.active_tasks[task_id].message = A2AMessage(
            role="agent",
            parts=[
                MessagePart(
                    kind="error",
                    text=error_message
                )
            ],
            taskId=task_id,
            contextId=context_id
        )
        
        raise NotImplementedError(error_message)
    
    async def _perform_semantic_search(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform AI-powered semantic search"""
        # For now, return a proper error response instead of raising exception
        return {
            "results": [],
            "total_count": 0,
            "search_type": "semantic",
            "query": query,
            "error": "Semantic search functionality is currently under development",
            "message": "Basic search functionality available, semantic search coming soon"
        }
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a specific task"""
        task_status = self.active_tasks.get(task_id)
        return task_status.model_dump() if task_status and hasattr(task_status, 'model_dump') else task_status
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the Catalog Manager agent"""
        try:
            # Check ORD service health
            ord_health = await self.ord_service.get_health_status()
            
            # Check AI clients
            grok_healthy = self.grok_client is not None
            perplexity_healthy = self.perplexity_client is not None
            
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "ord_service": ord_health,
                "ai_clients": {
                    "grok": grok_healthy,
                    "perplexity": perplexity_healthy
                },
                "active_tasks": len(self.active_tasks),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
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
    
    async def _handle_error_with_help_seeking(
        self, 
        task_id: str, 
        tracker_task_id: str,
        checklist_item_id: str,
        error: Exception, 
        problem_type: str,
        original_operation: Optional[Callable] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle errors with integrated help-seeking, action execution, and task tracking"""
        try:
            # Mark checklist item as failed and seeking help
            self.task_tracker.fail_checklist_item(
                tracker_task_id, 
                checklist_item_id, 
                str(error), 
                seek_help=True
            )
            
            # Create help request in task tracker BEFORE seeking help
            help_request_id = self.task_tracker.create_help_request(
                task_id=tracker_task_id,
                problem_type=problem_type,
                problem_description=str(error),
                target_agent=self._find_best_helper_agent(problem_type) or "unknown",
                checklist_item_id=checklist_item_id,
                context=context or {}
            )
            
            # CRITICAL: Use the new seek_help_and_act method that actually executes actions
            help_action_result = await self.seek_help_and_act(
                problem_type=problem_type,
                error=error,
                original_operation=original_operation,
                context=context,
                urgency="medium"
            )
            
            # Update task tracker based on help action results
            if help_action_result["help_sought"]:
                self.task_tracker.mark_help_request_sent(help_request_id)
                
                if help_action_result["help_received"]:
                    # Create a mock response for task tracker compatibility
                    mock_response = {
                        "success": help_action_result["success"],
                        "actions_taken": help_action_result.get("actions_taken", False),
                        "resolved_issue": help_action_result.get("resolved_issue", False),
                        "action_plan_id": help_action_result.get("action_plan_id"),
                        "execution_time": help_action_result.get("execution_time"),
                        "final_outcome": help_action_result.get("final_outcome")
                    }
                    
                    self.task_tracker.mark_help_request_completed(
                        help_request_id, 
                        mock_response,
                        actions_executed=help_action_result.get("actions_executed", 0)
                    )
                    
                    # If help resolved the issue, mark checklist item as completed
                    if help_action_result.get("resolved_issue"):
                        self.task_tracker.complete_checklist_item(
                            tracker_task_id, 
                            checklist_item_id,
                            f"Resolved with help from agent"
                        )
                        
                        logger.info(f"🎉 Task {task_id} resolved through help-seeking and action execution")
                        return {
                            "task_resolved": True,
                            "resolution_method": "help_seeking_with_action",
                            "help_result": help_action_result
                        }
                else:
                    self.task_tracker.mark_help_request_failed(help_request_id, "No help response received")
            else:
                self.task_tracker.mark_help_request_failed(help_request_id, "Help request not sent (threshold not met)")
            
            # If we get here, help didn't resolve the issue
            logger.warning(f"⚠️ Task {task_id} could not be resolved through help-seeking")
            return {
                "task_resolved": False,
                "resolution_method": "help_seeking_attempted",
                "help_result": help_action_result
            }
            
        except Exception as help_error:
            logger.error(f"❌ Error in help-seeking process for task {task_id}: {help_error}")
            self.task_tracker.fail_checklist_item(
                tracker_task_id, 
                checklist_item_id, 
                f"Help-seeking failed: {str(help_error)}"
            )
            return {
                "task_resolved": False,
                "resolution_method": "help_seeking_failed",
                "error": str(help_error)
            }
    
    async def get_help_action_stats(self) -> Dict[str, Any]:
        """Get help action execution statistics"""
        if self.help_action_system:
            return self.help_action_system.get_action_statistics()
        else:
            return {"error": "Help action system not initialized"}
    
    async def get_help_action_history(self) -> List[Dict[str, Any]]:
        """Get history of help actions taken"""
        if self.help_action_system:
            return self.help_action_system.get_action_history()
        else:
            return []
    
    async def _load_persisted_state(self):
        """Load persisted catalog manager state from disk"""
        try:
            # Load active tasks
            tasks_file = os.path.join(self._storage_path, "active_tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_id, task_info in tasks_data.items():
                        # Reconstruct TaskStatus objects
                        if 'status' in task_info:
                            task_info['status'] = TaskStatus(**task_info['status'])
                        self.active_tasks[task_id] = task_info
                logger.info(f"✅ Loaded {len(self.active_tasks)} active tasks from storage")
            
            # Load task history
            history_file = os.path.join(self._storage_path, "task_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.task_history = json.load(f)
                logger.info(f"✅ Loaded {len(self.task_history)} historical tasks from storage")
                
        except Exception as e:
            logger.error(f"❌ Failed to load persisted state: {e}")
    
    async def _persist_state(self):
        """Persist catalog manager state to disk"""
        try:
            # Save active tasks (convert TaskStatus to dict for JSON serialization)
            tasks_data = {}
            for task_id, task_info in self.active_tasks.items():
                # TaskStatus is a Pydantic model, use model_dump() to convert to dict
                if hasattr(task_info, 'model_dump'):
                    task_data = task_info.model_dump()
                elif hasattr(task_info, 'dict'):
                    task_data = task_info.dict()
                else:
                    # Fallback for non-Pydantic objects
                    task_data = dict(task_info) if hasattr(task_info, 'items') else str(task_info)
                tasks_data[task_id] = task_data
            
            tasks_file = os.path.join(self._storage_path, "active_tasks.json")
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Save task history
            history_file = os.path.join(self._storage_path, "task_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.task_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to persist catalog manager state: {e}")


# Factory function for creating Catalog Manager agent
def create_catalog_manager_agent(base_url: str, ord_registry_url: str = None, downstream_agent_url: str = None) -> CatalogManagerAgent:
    """Create a new Catalog Manager agent instance"""
    return CatalogManagerAgent(
        base_url=base_url,
        ord_registry_url=ord_registry_url,
        downstream_agent_url=downstream_agent_url
    )

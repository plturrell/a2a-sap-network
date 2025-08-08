import asyncio
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
import logging
import httpx
import hashlib

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..core.a2a_types import A2AMessage, MessagePart, MessageRole
from ..agents.data_standardization_agent import (
    TaskState, TaskStatus, TaskArtifact, AgentCard
)
from app.a2a.core.workflow_context import workflow_context_manager, DataArtifact
from app.a2a.core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import initialize_agent_trust, sign_a2a_message, get_trust_contract, verify_a2a_message
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
# Registry client removed - using Agent Manager for service discovery
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus, ChecklistItem
from ..core.telemetry import (
    init_telemetry, instrument_httpx, trace_async, trace_agent_message,
    trace_agent_task, add_span_attributes, add_span_event, get_trace_context
)
from ..config.telemetry_config import telemetry_config
from ..sdk import A2AAgentBase, a2a_handler, a2a_skill, a2a_task
from ..sdk.utils import create_agent_id, create_success_response, create_error_response
from opentelemetry import trace

logger = logging.getLogger(__name__)


class DataProductRegistrationAgent(AgentHelpSeeker):
    """Agent 0: Data Product Registration Agent with Dublin Core - Enhanced metadata extraction and quality assessment"""
    
    def __init__(self, base_url: str, ord_registry_url: str, downstream_agent_url: str = None):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        self.ord_registry_url = ord_registry_url
        self.downstream_agent_url = downstream_agent_url  # Can be None for dynamic discovery
        # Registry client removed - using Agent Manager for service discovery
        
        # Agent identification (trust identity will be registered via Agent Manager)
        self.agent_id = "data_product_agent_0"
        self.agent_name = "DataProductRegistrationAgent"
        self.agent_identity = None  # Will be set after Agent Manager registration
        logger.info(f"Agent 0 identity pending registration via Agent Manager: {self.agent_id}")
        
        # Initialize OpenTelemetry
        if telemetry_config.otel_enabled:
            self.tracer = init_telemetry(
                service_name=f"a2a-agent-{self.agent_id}",
                agent_id=self.agent_id,
                sampling_rate=telemetry_config.otel_traces_sampler_arg
            )
            instrument_httpx()  # Instrument HTTP client
            logger.info(f"OpenTelemetry initialized for {self.agent_id}")
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name="Data Product Registration Agent"
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "ord_registry_url": self.ord_registry_url,
            "agent_type": "data_product_registration",
            "timeout": 30.0,
            "retry.max_attempts": 3
        }
        self.initialize_help_action_system(self.agent_id, agent_context)
        
        self.agent_card = AgentCard(
            name="Data Product Registration Agent with Dublin Core",
            description="A2A v0.2.9 compliant agent that processes raw data into CDS schema with ORD descriptors enhanced by Dublin Core metadata, registers data via ORD references (no raw data transfer), and triggers downstream A2A agents",
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
                "metadataExtraction": True,
                "dublinCoreCompliance": True,
                "smartContractDelegation": True,
                "aiAdvisor": True,
                "helpSeeking": True,
                "taskTracking": True
            },
            defaultInputModes=["text/plain", "text/csv", "application/json", "application/xml"],
            defaultOutputModes=["application/json", "application/cds", "application/ord+json"],
            skills=[
                {
                    "id": "dublin-core-extraction",
                    "name": "Dublin Core Metadata Extraction",
                    "description": "Extract and generate Dublin Core metadata from raw data according to ISO 15836, RFC 5013, and ANSI/NISO Z39.85 standards",
                    "tags": ["dublin-core", "metadata", "iso15836", "rfc5013", "standards"],
                    "inputModes": ["text/csv", "application/json", "application/xml", "text/plain"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "cds-csn-generation",
                    "name": "CDS CSN Generation",
                    "description": "Generate Core Data Services CSN (Core Schema Notation) from raw financial data",
                    "tags": ["cds", "csn", "schema", "sap-cap"],
                    "inputModes": ["text/csv", "application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "ord-descriptor-creation-with-dublin-core",
                    "name": "ORD Descriptor Creation with Dublin Core",
                    "description": "Generate Object Resource Discovery descriptors enhanced with Dublin Core metadata for improved discoverability",
                    "tags": ["ord", "discovery", "metadata", "dublin-core"],
                    "inputModes": ["application/json", "application/cds"],
                    "outputModes": ["application/ord+json"]
                },
                {
                    "id": "catalog-registration-enhanced",
                    "name": "Enhanced Data Catalog Registration",
                    "description": "Register data products in enterprise data catalog with Dublin Core metadata",
                    "tags": ["catalog", "registration", "governance", "dublin-core"],
                    "inputModes": ["application/json", "application/ord+json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "metadata-quality-assessment",
                    "name": "Dublin Core Quality Assessment",
                    "description": "Assess and score Dublin Core metadata quality according to standards",
                    "tags": ["quality", "assessment", "dublin-core", "validation"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "a2a-orchestration",
                    "name": "A2A Downstream Triggering",
                    "description": "Trigger downstream A2A agents with catalog references and Dublin Core context",
                    "tags": ["a2a", "orchestration", "pipeline"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "smart-delegation",
                    "name": "Smart Contract Delegation",
                    "description": "Delegate tasks to Data Manager and Catalog Manager via smart contracts",
                    "tags": ["delegation", "smart-contract", "a2a"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "ai-advisor",
                    "name": "AI-Powered Help and Guidance",
                    "description": "Intelligent advisor for troubleshooting and guidance using Grok-4",
                    "tags": ["ai", "advisor", "help", "grok-4"],
                    "inputModes": ["text/plain", "application/json"],
                    "outputModes": ["application/json"]
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
            learning_threshold=8,  # Slightly higher threshold for data product decisions
            cache_ttl=300
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI Advisor
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name="Data Product Registration Agent",
            agent_capabilities={
                "streaming": True,
                "pushNotifications": True,
                "stateTransitionHistory": True,
                "batchProcessing": True,
                "metadataExtraction": True,
                "dublinCoreCompliance": True,
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
            max_concurrent_processing=3,  # Conservative for data processing
            auto_mode_threshold=5,        # Switch to queue after 5 pending
            enable_streaming=True,        # Support real-time processing
            enable_batch_processing=True  # Support batch processing
        )
        
        self.tasks = {}
        self.cancelled_tasks = set()
        
        # Initialize persistent storage for tasks and products  
        self._storage_path = os.getenv("DATA_PRODUCT_AGENT_STORAGE_PATH", "/tmp/data_product_agent_state")
        os.makedirs(self._storage_path, exist_ok=True)
        
        # Load persisted state
        asyncio.create_task(self._load_persisted_state())
        
        # Set message queue processor callback after all methods are defined
        if self.message_queue:
            self.message_queue.set_message_processor(self._process_message_core)
        
        logger.info("✅ Agent 0 v2.0.0 initialized with AI advisor, delegation, help-seeking, and message queue capabilities")
    
    def _initialize_advisor_knowledge(self):
        """Initialize AI advisor with agent-specific knowledge"""
        # Add FAQs
        self.ai_advisor.add_faq_item(
            "What does Agent 0 do?",
            "I am the Data Product Registration Agent (A2A v0.2.9 compliant). I process raw financial data, extract Dublin Core metadata, create CDS schemas, generate ORD descriptors, and register data products in the catalog using ORD references only. I delegate data storage to the Data Manager and metadata registration to the Catalog Manager without transferring raw data, following A2A protocol v0.2.9 specifications."
        )
        
        self.ai_advisor.add_faq_item(
            "What agents do you work with?",
            "I work closely with the Data Manager Agent (for data storage operations) and the Catalog Manager Agent (for metadata registration and enhancement). I also trigger the Financial Standardization Agent (Agent 1) after registration."
        )
        
        self.ai_advisor.add_faq_item(
            "What data formats do you support?",
            "I support CSV, JSON, XML, and plain text formats. I'm specifically designed for CRD financial data extractions and can handle large datasets with integrity verification."
        )
        
        # Add common issues
        self.ai_advisor.add_common_issue(
            "data_location_not_found",
            "Raw data files are not found at the specified location",
            "Check that the data_location path is correct and files exist. Default location is /Users/apple/projects/finsight_cib/data/raw. Verify file permissions and ensure CRD_Extraction_*.csv files are present."
        )
        
        self.ai_advisor.add_common_issue(
            "ord_registration_failed",
            "ORD registry registration fails with timeout or error",
            "Verify ORD registry is running on the configured URL. Check network connectivity. Ensure the ORD document is properly formatted. Try reducing the payload size if timeout occurs."
        )
        
        self.ai_advisor.add_common_issue(
            "delegation_denied",
            "Smart contract delegation to Data Manager or Catalog Manager is denied",
            "Check trust scores between agents. Verify both target agents are registered in the trust contract. Ensure delegation permissions are properly configured in the delegation contract."
        )
        
        # Add help-seeking knowledge
        self.ai_advisor.add_faq_item(
            "Can you ask other agents for help?",
            "Yes! I can actively seek help from other agents when I encounter problems. I know how to contact the Data Manager for storage issues, the Financial Standardization Agent for data processing problems, and the Catalog Manager for metadata issues."
        )
        
        self.ai_advisor.add_common_issue(
            "help_seeking_available",
            "When should I seek help from other agents?",
            "I automatically seek help when I encounter repeated failures (3+ consecutive errors), network connectivity issues, service unavailability, or data integrity problems. I can ask the Data Manager for storage help, Agent 1 for standardization guidance, and others based on the problem type."
        )
        
        # Add operational state inquiry knowledge
        self.ai_advisor.add_faq_item(
            "What are you currently doing?",
            "I can report my current operational state including any active tasks, their progress, help requests status, and overall processing state. Use my get_agent_operational_state() method or ask me directly about my current status."
        )
        
        self.ai_advisor.add_faq_item(
            "What processes are you running?",
            "I track all my active tasks with detailed progress including checklist completion, any pending help requests, and current processing stage. I can tell you about data registration workflows, ORD creation processes, and any help-seeking activities."
        )
        
        self.ai_advisor.add_faq_item(
            "How can I check your status?",
            "You can ask me 'what are you doing?', 'what processes are running?', or 'what is your current state?' through A2A messages. I'll provide detailed information about my operational state, active tasks, and any issues I'm experiencing."
        )
    
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
                        "final_outcome": help_action_result.get("final_outcome")
                    }
                    self.task_tracker.receive_help_response(help_request_id, mock_response)
                    
                    if help_action_result.get("actions_taken", False):
                        # Determine effectiveness based on whether issue was resolved
                        if help_action_result.get("resolved_issue", False):
                            effectiveness = 5  # Excellent - issue resolved
                            notes = f"Help resolved issue. Actions executed: {help_action_result.get('actions_executed', 0)}"
                            
                            # Mark checklist item as completed since issue is resolved
                            self.task_tracker.complete_checklist_item(
                                tracker_task_id, 
                                checklist_item_id, 
                                notes
                            )
                        else:
                            effectiveness = 3  # Moderate - actions taken but issue not fully resolved
                            notes = f"Help actions taken but issue not fully resolved. Actions executed: {help_action_result.get('actions_executed', 0)}"
                        
                        self.task_tracker.apply_help_solution(help_request_id, effectiveness, notes)
                        
                        logger.info(f"✅ Help actions executed for {problem_type}. Resolved: {help_action_result.get('resolved_issue', False)}")
                    else:
                        # Help received but no actions could be taken
                        effectiveness = 2
                        self.task_tracker.apply_help_solution(help_request_id, effectiveness, "Help received but no actions could be executed")
                        
                        logger.warning(f"⚠️ Help received for {problem_type} but no actions could be taken")
            
            return help_action_result
                
        except Exception as help_error:
            logger.error(f"❌ Error in help-seeking and action process: {help_error}")
            return {
                "success": False,
                "help_sought": False,
                "error": str(help_error),
                "resolved_issue": False
            }
    
    def _get_checklist_item_id_by_description(self, tracker_task_id: str, description_keyword: str) -> Optional[str]:
        """Get checklist item ID by description keyword"""
        if tracker_task_id in self.task_tracker.tasks:
            task = self.task_tracker.tasks[tracker_task_id]
            for item in task.checklist:
                if description_keyword.lower() in item.description.lower():
                    return item.item_id
        return None
    
    def _calculate_row_hash(self, row_data: Dict) -> str:
        """Calculate SHA256 hash for a data row"""
        # Create a stable string representation of the row
        row_str = json.dumps(row_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(row_str.encode('utf-8')).hexdigest()
    
    def _calculate_dataset_checksum(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate checksums and integrity info for a dataset"""
        # For ORD registry, we only store summary statistics, not individual row hashes
        # Individual row verification can be done when data is staged in database
        
        # Calculate dataset hash from row count and first/last row samples for large datasets
        if data:
            first_row_hash = self._calculate_row_hash(data[0])
            last_row_hash = self._calculate_row_hash(data[-1]) if len(data) > 1 else first_row_hash
            summary_string = f"{len(data)}|{first_row_hash}|{last_row_hash}"
        else:
            summary_string = "0||"
        
        combined_hash = hashlib.sha256(summary_string.encode('utf-8')).hexdigest()
        
        return {
            "row_count": len(data),
            "dataset_hash": combined_hash,
            "first_row_hash": first_row_hash if data else "",
            "last_row_hash": last_row_hash if data else "",
            "timestamp": datetime.utcnow().isoformat(),
            "staging_recommended": len(data) > 1000  # Flag large datasets for database staging
        }
    
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
            # Process help request through AI advisor
            asking_agent_id = getattr(message, 'from_agent_id', None)
            advisor_response = await self.ai_advisor.process_a2a_help_message(
                [part.dict() for part in message.parts],
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
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card"""
        return self.agent_card.dict()
    
    async def _register_with_agent_manager(self) -> Dict[str, Any]:
        """Register this agent with the Agent Manager for ecosystem coordination"""
        try:
            # Create A2A message for agent registration
            registration_message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="data",
                        data={
                            "operation": "register_agent",
                            "agent_id": self.agent_id,
                            "agent_name": self.agent_name,
                            "base_url": self.base_url,
                            "capabilities": self.agent_card.capabilities,
                            "skills": self.agent_card.skills,
                            "metadata": {
                                "specialization": "data_product_registration",
                                "version": "2.0.0",
                                "protocol_version": "0.2.9"
                            }
                        }
                    )
                ]
            )
            
            # Get Agent Manager URL from configuration or discovery
            agent_manager_url = os.getenv("AGENT_MANAGER_URL")
            if not agent_manager_url:
                return {"success": False, "error": "Agent Manager URL not configured - set AGENT_MANAGER_URL environment variable"}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    agent_manager_url,
                    json={
                        "message": registration_message.model_dump(),
                        "contextId": f"registration_{self.agent_id}",
                        "priority": "high",
                        "processing_mode": "immediate"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Agent 0 registered with Agent Manager")
                    return result
                else:
                    logger.error(f"Failed to register with Agent Manager: {response.status_code}")
                    return {"success": False, "error": f"Registration failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error registering with Agent Manager: {e}")
            return {"success": False, "error": str(e)}
    
    async def _request_trust_contract(self, delegate_agent: str, actions: List[str]) -> Dict[str, Any]:
        """Request trust contract creation via Agent Manager"""
        try:
            # Create A2A message for trust contract request
            contract_message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="data",
                        data={
                            "operation": "create_trust_contract",
                            "delegator_agent": self.agent_id,
                            "delegate_agent": delegate_agent,
                            "actions": actions,
                            "expiry_hours": 168,  # 1 week
                            "conditions": {
                                "purpose": "A2A data processing delegation",
                                "scope": "data_product_registration_workflow"
                            }
                        }
                    )
                ]
            )
            
            # Get Agent Manager URL from configuration 
            agent_manager_url = os.getenv("AGENT_MANAGER_URL")
            if not agent_manager_url:
                return {"success": False, "error": "Agent Manager URL not configured - set AGENT_MANAGER_URL environment variable"}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    agent_manager_url,
                    json={
                        "message": contract_message.model_dump(),
                        "contextId": f"trust_contract_{self.agent_id}_{delegate_agent}",
                        "priority": "high"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Trust contract requested for {delegate_agent}")
                    return result
                else:
                    logger.error(f"Failed to create trust contract: {response.status_code}")
                    return {"success": False, "error": f"Trust contract failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error requesting trust contract: {e}")
            return {"success": False, "error": str(e)}
    
    async def _discover_agents_via_manager(self, skills: Optional[List[str]] = None) -> Dict[str, Any]:
        """Discover agents via Agent Manager instead of direct registry access"""
        try:
            # Create A2A message for agent discovery
            discovery_message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="data",
                        data={
                            "operation": "discover_agents",
                            "skills": skills,
                            "requesting_agent": self.agent_id
                        }
                    )
                ]
            )
            
            # Get Agent Manager URL from configuration 
            agent_manager_url = os.getenv("AGENT_MANAGER_URL")
            if not agent_manager_url:
                return {"success": False, "error": "Agent Manager URL not configured - set AGENT_MANAGER_URL environment variable"}
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    agent_manager_url,
                    json={
                        "message": discovery_message.model_dump(),
                        "contextId": f"discovery_{self.agent_id}",
                        "priority": "medium"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Agent discovery completed via Agent Manager")
                    return result
                else:
                    logger.error(f"Failed to discover agents: {response.status_code}")
                    return {"success": False, "error": f"Discovery failed: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            return {"success": False, "error": str(e)}
    
    @trace_async("process_message", kind=trace.SpanKind.SERVER)
    async def process_message(
        self, 
        message: A2AMessage, 
        context_id: str,
        priority: str = "medium",
        processing_mode: str = "auto"
    ) -> Dict[str, Any]:
        """Process A2A message with queue support (streaming or batched)"""
        
        # Add trace context
        add_span_attributes({
            "agent.id": self.agent_id,
            "message.id": message.messageId,
            "message.role": message.role.value,
            "context.id": context_id,
            "priority": priority,
            "processing_mode": processing_mode
        })
        
        if not self.message_queue:
            # Fallback to direct processing if queue not initialized
            return await self._process_message_core(message, context_id)
        
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
    
    @trace_async("_process_message_core", kind=trace.SpanKind.INTERNAL)
    async def _process_message_core(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Process incoming A2A message with task tracking and help-seeking support"""
        task_id = str(uuid4())
        
        # Add tracing context
        add_span_attributes({
            "task.id": task_id,
            "message.parts_count": len(message.parts),
            "agent.name": self.agent_name
        })
        
        # Check if this is an AI advisor request
        if self._is_advisor_request(message):
            return await self._handle_advisor_request(message, context_id)
        
        # Extract workflow context from message if available
        workflow_context = None
        workflow_id = None
        
        # Check if this is part of an existing workflow
        for part in message.parts:
            if part.kind == "data" and "workflow_context" in part.data:
                wf_data = part.data["workflow_context"]
                workflow_id = wf_data.get("workflow_id")
                workflow_context = workflow_context_manager.get_context(workflow_id)
                break
        
        # If no workflow context, check if we should create one
        if not workflow_context:
            # Check if message requests workflow creation
            for part in message.parts:
                if part.kind == "data" and part.data.get("create_workflow"):
                    workflow_data = part.data.get("workflow_metadata", {})
                    workflow_context = workflow_context_manager.create_workflow_context(
                        workflow_plan_id=workflow_data.get("plan_id", "data_registration_plan"),
                        workflow_name=workflow_data.get("name", "Data Product Registration Workflow"),
                        initiated_by=self.agent_card.name,
                        trust_contract_id=workflow_data.get("trust_contract_id"),
                        sla_id=workflow_data.get("sla_id"),
                        required_trust_level=workflow_data.get("required_trust_level", 0.0),
                        initial_stage="data_ingestion",
                        metadata={
                            "source": "agent0",
                            "task_id": task_id
                        }
                    )
                    workflow_id = workflow_context.workflow_id
                    
                    # Start workflow monitoring
                    await workflow_monitor.start_workflow_monitoring(
                        workflow_context,
                        total_stages=3  # ingestion, registration, standardization
                    )
                    break
        
        # Initialize task with workflow context
        self.tasks[task_id] = {
            "taskId": task_id,
            "contextId": context_id,
            "workflowId": workflow_id,
            "status": TaskStatus(state=TaskState.PENDING),
            "artifacts": [],
            "events": []
        }
        
        # Persist state after task creation
        asyncio.create_task(self._persist_state())
        
        # Create comprehensive task in task tracker
        task_name = "Data Product Registration"
        task_description = "Process raw data, extract Dublin Core metadata, create CDS schema, and register with ORD"
        checklist_items = [
            "Analyze raw data structure and quality",
            "Extract Dublin Core metadata from raw data",
            "Generate CDS CSN schema definition",
            "Create ORD descriptor with Dublin Core enhancement",
            "Register data product in catalog",
            "Trigger downstream agent processing"
        ]
        
        tracker_task_id = self.task_tracker.create_task(
            name=task_name,
            description=task_description,
            checklist_items=checklist_items,
            priority=TaskPriority.MEDIUM,
            context_id=context_id,
            tags=["data-registration", "dublin-core", "ord"]
        )
        
        # Link the task tracker task to the traditional task
        self.tasks[task_id]["tracker_task_id"] = tracker_task_id
        self.task_tracker.start_task(tracker_task_id)
        
        # Persist state after linking tracker task
        asyncio.create_task(self._persist_state())
        
        # Start processing in background
        asyncio.create_task(self._execute_data_product_registration(
            task_id, message, context_id, workflow_context
        ))
        
        return {
            "taskId": task_id,
            "contextId": context_id,
            "workflowId": workflow_id,
            "status": self.tasks[task_id]["status"].dict()
        }
    
    async def _execute_data_product_registration(
        self, 
        task_id: str, 
        message: A2AMessage, 
        context_id: str,
        workflow_context=None
    ):
        """Execute the enhanced data product registration process with Dublin Core"""
        try:
            await self._update_status(task_id, TaskState.WORKING, "Analyzing raw data...")
            
            # Extract data location from message
            data_location = self._extract_data_location(message)
            
            # Step 1: Analyze raw data
            await self._update_status(task_id, TaskState.WORKING, "Analyzing data structure...")
            data_analysis = await self._analyze_raw_data(data_location)
            
            # Create data artifact for raw data if in workflow
            if workflow_context:
                raw_data_artifact = workflow_context_manager.create_data_artifact(
                    workflow_id=workflow_context.workflow_id,
                    artifact_type="raw_data",
                    location=data_location,
                    created_by=self.agent_card.name,
                    metadata={
                        "files_count": len(data_analysis.get("data_files", [])),
                        "total_records": data_analysis.get("total_records", 0),
                        "data_types": data_analysis.get("data_types", [])
                    }
                )
            
            # Step 2: Extract Dublin Core metadata
            await self._update_status(task_id, TaskState.WORKING, "Extracting Dublin Core metadata...")
            dublin_core_metadata = await self._extract_dublin_core_metadata(data_analysis)
            
            # Step 3: Assess Dublin Core quality
            await self._update_status(task_id, TaskState.WORKING, "Assessing metadata quality...")
            dublin_core_quality = await self._assess_dublin_core_quality(dublin_core_metadata)
            
            # Step 4: Enhance Dublin Core if needed
            if dublin_core_quality["overall_score"] < 0.6:
                await self._update_status(task_id, TaskState.WORKING, "Enhancing Dublin Core metadata...")
                dublin_core_metadata = await self._enhance_dublin_core_metadata(
                    dublin_core_metadata, dublin_core_quality, data_analysis
                )
                # Re-assess after enhancement
                dublin_core_quality = await self._assess_dublin_core_quality(dublin_core_metadata)
            
            # Step 5: Stage data to database for large datasets
            await self._update_status(task_id, TaskState.WORKING, "Staging large datasets to database...")
            staging_info = await self._stage_data_to_database(data_analysis)
            
            # Step 6: Generate CDS CSN
            await self._update_status(task_id, TaskState.WORKING, "Generating CDS Core Schema Notation...")
            cds_csn = await self._generate_cds_csn(data_analysis)
            
            # Step 7: Generate Enhanced ORD Descriptors with Dublin Core and staging info
            await self._update_status(task_id, TaskState.WORKING, "Creating enhanced ORD descriptors with Dublin Core...")
            ord_descriptors = await self._generate_ord_descriptors(data_analysis, cds_csn, dublin_core_metadata, staging_info)
            
            # Step 8: Register in Data Catalog with Dublin Core metadata
            await self._update_status(task_id, TaskState.WORKING, "Registering in data catalog with enhanced metadata...")
            catalog_registration = await self._register_in_catalog(ord_descriptors)
            
            # Create artifact for registered data product if in workflow
            if workflow_context:
                registered_artifact = workflow_context_manager.create_data_artifact(
                    workflow_id=workflow_context.workflow_id,
                    artifact_type="registered_data_product",
                    location=catalog_registration.get("registry_url", ""),
                    created_by=self.agent_card.name,
                    metadata={
                        "registration_id": catalog_registration.get("registration_id"),
                        "dublin_core_quality_score": dublin_core_quality["overall_score"],
                        "ord_descriptors": ord_descriptors
                    },
                    parent_artifact_ids=[raw_data_artifact.artifact_id] if 'raw_data_artifact' in locals() else []
                )
                
                # Update workflow stage
                await workflow_monitor.update_workflow_stage(
                    workflow_context.workflow_id,
                    "data_standardization",
                    self.agent_card.name
                )
            
            # Step 9: Trigger downstream agent with Dublin Core context
            await self._update_status(task_id, TaskState.WORKING, "Triggering standardization agent with Dublin Core context...")
            downstream_trigger = await self._trigger_downstream_agent(
                catalog_registration, context_id, dublin_core_quality, workflow_context
            )
            
            # Create final artifact with Dublin Core information
            artifact = TaskArtifact(
                name="Enhanced Data Product Registration Results",
                description="Complete data product registration with Dublin Core metadata and catalog reference",
                parts=[MessagePart(
                    kind="data",
                    data={
                        "data_product": {
                            "id": catalog_registration.get("registration_id"),
                            "name": "financial_data_products",
                            "catalog_url": catalog_registration.get("registry_url")
                        },
                        "dublin_core_metadata": dublin_core_metadata,
                        "dublin_core_quality": dublin_core_quality,
                        "cds_csn": cds_csn,
                        "ord_descriptor": ord_descriptors,
                        "catalog_registration": catalog_registration,
                        "downstream_trigger": downstream_trigger
                    }
                )]
            )
            
            self.tasks[task_id]["artifacts"].append(artifact)
            
            await self._update_status(
                task_id,
                TaskState.COMPLETED,
                f"Successfully registered data products with Dublin Core (quality score: {dublin_core_quality['overall_score']:.2f}) and triggered standardization"
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in enhanced data product registration: {str(e)}")
            logger.error(f"Full traceback: {error_details}")
            await self._update_status(
                task_id,
                TaskState.FAILED,
                error={
                    "code": "ENHANCED_REGISTRATION_ERROR",
                    "message": str(e),
                    "traceback": error_details
                }
            )
    
    def _extract_data_location(self, message: A2AMessage) -> str:
        """Extract data location from message"""
        for part in message.parts:
            if part.kind == "data":
                if "data_location" in part.data:
                    return part.data["data_location"]
                elif "processing_instructions" in part.data:
                    # Check if data location is embedded in instructions
                    instructions = part.data.get("processing_instructions", {})
                    if isinstance(instructions, dict):
                        data_path = os.getenv("DATA_RAW_PATH", "/tmp/data/raw")
                        if not os.path.exists(data_path):
                            raise ValueError(f"Data path does not exist: {data_path}. Set DATA_RAW_PATH environment variable.")
                        return data_path
            elif part.kind == "text":
                # Don't use text content as file path
                pass
        
        # Default to our known location
        data_path = os.getenv("DATA_RAW_PATH", "/tmp/data/raw")
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}. Set DATA_RAW_PATH environment variable.")
        return data_path
    
    async def _analyze_raw_data(self, data_location: str) -> Dict[str, Any]:
        """Analyze raw data files and extract structure"""
        analysis = {
            "data_files": [],
            "total_records": 0,
            "data_types": [],
            "referential_integrity": {}
        }
        
        # Scan for CSV files
        for filename in os.listdir(data_location):
            if filename.endswith('.csv') and filename.startswith('CRD_'):
                file_path = os.path.join(data_location, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Extract data type from filename
                    data_type = filename.replace('CRD_Extraction_v1_', '').replace('_sorted.csv', '')
                    
                    # Convert to list of dicts for processing
                    data_records = df.to_dict('records')
                    
                    # Calculate integrity info
                    integrity_info = self._calculate_dataset_checksum(data_records)
                    
                    file_info = {
                        "filename": filename,
                        "path": file_path,
                        "data_type": data_type,
                        "records": len(df),
                        "columns": list(df.columns),
                        "sample_data": df.head(3).to_dict('records'),
                        "integrity": integrity_info
                    }
                    
                    logger.info(f"Loaded {data_type} data: {integrity_info['row_count']} rows, hash: {integrity_info['dataset_hash'][:16]}...")
                    
                    analysis["data_files"].append(file_info)
                    analysis["total_records"] += len(df)
                    analysis["data_types"].append(data_type)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {filename}: {str(e)}")
        
        # Perform referential integrity analysis
        analysis["referential_integrity"] = await self._verify_referential_integrity(data_location, analysis)
        
        return analysis
    
    async def _verify_referential_integrity(self, data_location: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Verify referential integrity between transactional and dimensional data"""
        integrity_report = {
            "verification_timestamp": datetime.utcnow().isoformat(),
            "overall_status": "verified",
            "dimension_tables": {},
            "foreign_key_checks": {},
            "orphaned_records": {},
            "missing_references": {},
            "summary": {
                "total_fk_relationships": 0,
                "verified_relationships": 0,
                "broken_relationships": 0,
                "integrity_score": 0.0
            }
        }
        
        try:
            # Load the main transactional file (CRD_Extraction_Indexed.csv)
            indexed_file_path = os.path.join(data_location, "CRD_Extraction_Indexed.csv")
            if not os.path.exists(indexed_file_path):
                logger.warning("Main transactional file CRD_Extraction_Indexed.csv not found")
                return integrity_report
                
            logger.info("Loading transactional data for referential integrity verification...")
            df_main = pd.read_csv(indexed_file_path)
            
            # Load dimensional tables
            dimension_data = {}
            for file_info in analysis["data_files"]:
                if file_info["data_type"] != "CRD_Extraction_Indexed.csv":
                    df_dim = pd.read_csv(file_info["path"])
                    dimension_data[file_info["data_type"]] = df_dim
                    integrity_report["dimension_tables"][file_info["data_type"]] = {
                        "records": len(df_dim),
                        "columns": list(df_dim.columns),
                        "id_column": self._identify_id_column(df_dim, file_info["data_type"])
                    }
            
            # Define foreign key relationships based on column names
            fk_relationships = {
                "books_id": {"table": "book", "column": "_row_number"},
                "location_id": {"table": "location", "column": "_row_number"}, 
                "account_id": {"table": "account", "column": "_row_number"},
                "product_id": {"table": "product", "column": "_row_number"},
                "measure_id": {"table": "measure", "column": "_row_number"}
            }
            
            # Verify each foreign key relationship
            for fk_column, reference in fk_relationships.items():
                if fk_column in df_main.columns:
                    ref_table = reference["table"]
                    ref_column = reference["column"]
                    
                    if ref_table in dimension_data:
                        integrity_report["summary"]["total_fk_relationships"] += 1
                        
                        # Get unique foreign key values from main table
                        fk_values = set(df_main[fk_column].dropna().astype(int))
                        
                        # Get available primary key values from dimension table
                        dim_df = dimension_data[ref_table]
                        if ref_column in dim_df.columns:
                            pk_values = set(dim_df[ref_column].dropna().astype(int))
                        else:
                            # Fallback: use index + 1 as ID
                            pk_values = set(range(1, len(dim_df) + 1))
                        
                        # Find orphaned records (FK values with no matching PK)
                        orphaned = fk_values - pk_values
                        missing_refs = pk_values - fk_values
                        
                        # Calculate integrity metrics
                        total_fk_records = len(df_main[df_main[fk_column].notna()])
                        orphaned_count = len(df_main[df_main[fk_column].isin(orphaned)])
                        
                        fk_check = {
                            "foreign_key_column": fk_column,
                            "reference_table": ref_table,
                            "reference_column": ref_column,
                            "total_fk_records": total_fk_records,
                            "unique_fk_values": len(fk_values),
                            "available_pk_values": len(pk_values),
                            "orphaned_fk_values": len(orphaned),
                            "orphaned_records_count": orphaned_count,
                            "missing_references_count": len(missing_refs),
                            "integrity_ratio": (total_fk_records - orphaned_count) / total_fk_records if total_fk_records > 0 else 0.0,
                            "status": "verified" if len(orphaned) == 0 else "integrity_violations"
                        }
                        
                        if len(orphaned) == 0:
                            integrity_report["summary"]["verified_relationships"] += 1
                            logger.info(f"✅ FK integrity verified: {fk_column} -> {ref_table}.{ref_column}")
                        else:
                            integrity_report["summary"]["broken_relationships"] += 1
                            logger.warning(f"❌ FK integrity violation: {fk_column} -> {ref_table}.{ref_column} ({len(orphaned)} orphaned values)")
                            integrity_report["orphaned_records"][fk_column] = list(orphaned)
                        
                        if len(missing_refs) > 0:
                            integrity_report["missing_references"][ref_table] = list(missing_refs) 
                            logger.info(f"ℹ️  Unused dimension records in {ref_table}: {len(missing_refs)} records")
                        
                        integrity_report["foreign_key_checks"][fk_column] = fk_check
                    else:
                        logger.warning(f"Referenced dimension table '{ref_table}' not found for FK {fk_column}")
                else:
                    logger.warning(f"Foreign key column '{fk_column}' not found in main table")
            
            # Calculate overall integrity score
            total_relationships = integrity_report["summary"]["total_fk_relationships"]
            if total_relationships > 0:
                integrity_report["summary"]["integrity_score"] = integrity_report["summary"]["verified_relationships"] / total_relationships
            
            # Set overall status
            if integrity_report["summary"]["broken_relationships"] == 0:
                integrity_report["overall_status"] = "verified"
                logger.info(f"🎉 Referential integrity VERIFIED: {integrity_report['summary']['integrity_score']:.2%} integrity score")
            else:
                integrity_report["overall_status"] = "violations_detected"
                logger.error(f"💥 Referential integrity VIOLATIONS detected: {integrity_report['summary']['broken_relationships']} broken relationships")
        
        except Exception as e:
            logger.error(f"Error during referential integrity verification: {str(e)}")
            integrity_report["overall_status"] = "verification_failed"
            integrity_report["error"] = str(e)
        
        return integrity_report
    
    def _identify_id_column(self, df: pd.DataFrame, table_name: str) -> str:
        """Identify the ID column in a dimension table"""
        # Check for common ID column patterns
        for col in df.columns:
            if col.lower() in ['id', '_row_number', 'row_number', f'{table_name}_id']:
                return col
        
        # If no explicit ID column, assume index-based (1-indexed)
        return "_index_plus_one"
    
    async def _stage_data_to_database(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Stage raw data to database for efficient access by downstream agents"""
        staging_info = {}
        
        try:
            # Check if Data Manager is available for A2A staging
            data_manager_available = await self._check_data_manager_availability()
            
            if not data_manager_available:
                logger.warning("Data Manager agent not available - using file-only staging")
                # Fall back to file-only staging for all datasets
                for file_info in data_analysis["data_files"]:
                    data_type = file_info["data_type"]
                    file_path = file_info["path"]
                    staging_info[data_type] = {
                        "staging_method": "file",
                        "access_strategy": {
                            "type": "file",
                            "path": file_path
                        },
                        "reason": "Data Manager agent not available"
                    }
                return staging_info
            
            for file_info in data_analysis["data_files"]:
                data_type = file_info["data_type"]
                file_path = file_info["path"]
                
                # Only skip very small datasets (< 100 records) for file access
                if file_info["records"] < 100:
                    staging_info[data_type] = {
                        "staging_method": "file",
                        "access_strategy": {
                            "type": "file",
                            "path": file_path
                        },
                        "reason": "Very small dataset - file access is efficient"
                    }
                    continue
                
                logger.info(f"Staging {data_type} data via Data Manager ({file_info['records']} rows)")
                
                # Read the data (handle large files efficiently)
                try:
                    if file_info["records"] > 100000:  # Large file - read in chunks
                        logger.info(f"Reading large file {file_path} in chunks")
                        df = pd.read_csv(file_path, chunksize=10000)
                        records = []
                        for chunk in df:
                            records.extend(chunk.to_dict('records'))
                    else:
                        df = pd.read_csv(file_path)
                        records = df.to_dict('records')
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {e}")
                    # Fallback to file access
                    staging_info[data_type] = {
                        "staging_method": "file",
                        "access_strategy": {
                            "type": "file",
                            "path": file_path
                        },
                        "reason": f"File reading failed: {str(e)}"
                    }
                    continue
                
                # Use A2A message to Data Manager for staging
                table_name = f"crd_{data_type}_data"
                storage_type = "hana" if file_info["records"] > 1000 else "supabase"
                
                staging_result = await self._stage_to_data_manager(
                    records=records,
                    table_name=table_name,
                    data_type=data_type,
                    file_info=file_info,
                    storage_type=storage_type
                )
                
                if staging_result.get("success"):
                    staging_info[data_type] = {
                        "staging_method": "database",
                        "access_strategy": {
                            "type": "a2a_database",
                            "agent": "data_manager",
                            "storage_type": storage_type,
                            "table": table_name,
                            "query_filter": {
                                "data_source": f"crd_{data_type}",
                                "data_type": data_type
                            }
                        },
                        "records_staged": staging_result.get("records_created", len(records)),
                        "staging_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"✅ Successfully staged {len(records)} {data_type} records via Data Manager")
                else:
                    # Fallback to file access if staging failed
                    staging_info[data_type] = {
                        "staging_method": "file",
                        "access_strategy": {
                            "type": "file",
                            "path": file_path
                        },
                        "reason": f"Data Manager staging failed: {staging_result.get('error', 'Unknown error')}"
                    }
                    logger.warning(f"⚠️ Data Manager staging failed for {data_type}, using file access")
        
        except Exception as e:
            logger.error(f"Error in database staging: {str(e)}")
            # Fallback to file-based access for all
            for file_info in data_analysis["data_files"]:
                data_type = file_info["data_type"]
                staging_info[data_type] = {
                    "staging_method": "file",
                    "access_strategy": {
                        "type": "file", 
                        "path": file_info["path"]
                    },
                    "reason": "Database staging not available"
                }
        
        return staging_info
    
    
    async def _generate_cds_csn(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CDS Core Schema Notation from data analysis"""
        definitions = {}
        
        for file_info in data_analysis["data_files"]:
            data_type = file_info["data_type"]
            columns = file_info["columns"]
            
            # Create entity definition
            entity_name = f"{data_type.capitalize()}Entity"
            
            elements = {}
            for col in columns:
                # Determine CDS type based on column name and sample data
                cds_type = self._infer_cds_type(col, file_info.get("sample_data", []))
                elements[col.replace(" ", "_").replace("(", "").replace(")", "")] = cds_type
            
            definitions[entity_name] = {
                "kind": "entity",
                "elements": elements
            }
        
        # Create complete CSN
        csn = {
            "definitions": definitions,
            "meta": {
                "creator": "DataProductRegistrationAgent",
                "flavor": "inferred",
                "namespace": "com.finsight.cib"
            },
            "$version": "2.0"
        }
        
        return csn
    
    def _infer_cds_type(self, column_name: str, sample_data: List[Dict]) -> Dict[str, Any]:
        """Infer CDS type from column name and sample data"""
        # Simple type inference
        if "id" in column_name.lower() or "_number" in column_name.lower():
            return {"type": "cds.Integer"}
        elif "date" in column_name.lower() or "time" in column_name.lower():
            return {"type": "cds.DateTime"}
        elif "amount" in column_name.lower() or "value" in column_name.lower():
            return {"type": "cds.Decimal", "precision": 15, "scale": 2}
        else:
            # Default to string
            return {"type": "cds.String", "length": 255}
    
    async def _extract_dublin_core_metadata(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dublin Core metadata from data analysis according to ISO 15836"""
        import os
        from datetime import datetime
        
        # Aggregate information from all data files
        all_data_types = data_analysis.get("data_types", [])
        total_records = data_analysis.get("total_records", 0)
        file_count = len(data_analysis.get("data_files", []))
        
        # Extract title from first file or generate
        title = "Financial Data Collection"
        if data_analysis.get("data_files"):
            first_file = data_analysis["data_files"][0]
            filename = first_file.get("filename", "")
            if filename:
                # Clean filename for title
                title = filename.replace("CRD_Extraction_v1_", "").replace("_sorted.csv", "").replace("_", " ").title()
                title = f"CRD {title} Financial Data"
        
        # Generate comprehensive description
        description = f"Financial data collection containing {total_records:,} records across {file_count} data types. "
        if all_data_types:
            description += f"Data types include: {', '.join(all_data_types)}."
        
        # Extract date from file timestamps or use current
        date = datetime.utcnow().isoformat()
        if data_analysis.get("data_files"):
            # Try to get file modification time
            try:
                first_file_path = data_analysis["data_files"][0].get("path")
                if first_file_path and os.path.exists(first_file_path):
                    mtime = os.path.getmtime(first_file_path)
                    date = datetime.fromtimestamp(mtime).isoformat()
            except:
                pass
        
        dublin_core = {
            "title": f"CRD Financial Data Products - {datetime.utcnow().strftime('%B %Y')}",
            "creator": ["FinSight CIB", "Data Product Registration Agent", "CRD System"],
            "subject": ["financial-data", "crd-extraction", "raw-data", "enterprise-data"] + all_data_types,
            "description": description,
            "publisher": "FinSight CIB Data Platform",
            "contributor": ["CRD Extraction Process", "Data Pipeline Team", "Financial Systems"],
            "date": date,
            "type": "Dataset",
            "format": "text/csv",
            "identifier": f"crd-financial-data-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "source": "CRD Financial System - Core Banking Platform",
            "language": "en",
            "relation": ["com.finsight.cib:pipeline:financial_standardization"],
            "coverage": f"Financial Period {datetime.utcnow().strftime('%Y-%m')}",
            "rights": "Internal Use Only - FinSight CIB Proprietary Data"
        }
        
        return dublin_core
    
    async def _check_data_manager_availability(self) -> bool:
        """Check if Data Manager agent is available for A2A communication"""
        try:
            # Use Agent Manager for agent discovery instead of direct registry access
            discovery_result = await self._discover_agents_via_manager(skills=["data-storage"])
            if not discovery_result.get("success", False):
                return False
            
            # Look for Data Manager agent in the discovered agents
            agents = discovery_result.get("agents", [])
            data_manager_agents = [agent for agent in agents if "data_manager" in agent.get("id", "").lower()]
            
            return len(data_manager_agents) > 0
        except Exception as e:
            logger.warning(f"Failed to check Data Manager availability: {e}")
            return False
    
    async def _stage_to_data_manager(self, records: List[Dict], table_name: str, data_type: str, 
                                    file_info: Dict, storage_type: str) -> Dict[str, Any]:
        """Stage data to database via A2A message to Data Manager"""
        try:
            # Create A2A message for data storage
            data_request = {
                "operation": "create",
                "path": table_name,
                "data": records,
                "storage_type": storage_type,
                "service_level": "silver",  # Standard processing
                "context": {
                    "source_agent": self.agent_id,
                    "data_type": data_type,
                    "source_file": file_info.get("path"),
                    "columns": file_info.get("columns", []),
                    "integrity": file_info.get("integrity", {}),
                    "staging_purpose": "data_product_registration"
                }
            }
            
            message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="data",
                        data=data_request
                    )
                ],
                taskId=str(uuid4()),
                contextId=str(uuid4())
            )
            
            # Send message to Data Manager via Agent Manager discovery
            discovery_result = await self._discover_agents_via_manager(skills=["data-storage"])
            if discovery_result.get("success", False):
                # Get Data Manager endpoint
                agents = discovery_result.get("agents", [])
                data_manager = next((agent for agent in agents if "data_manager" in agent.get("id", "").lower()), None)
                
                if data_manager:
                    data_manager_url = data_manager.get("url")
                    if not data_manager_url:
                        logger.error("Data Manager agent found but no URL provided")
                        return {"success": False, "error": "Data Manager URL not available"}
                    
                    # Send A2A message
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{data_manager_url}/a2a/data_manager/v1/messages",
                            json={
                                "message": message.model_dump(),
                                "contextId": message.contextId,
                                "priority": "medium"
                            },
                            timeout=60.0
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            return {"success": True, "result": result}
                        else:
                            logger.error(f"Data Manager staging failed: {response.status_code} - {response.text}")
                            return {"success": False, "error": f"HTTP {response.status_code}"}
            
            # Fallback if no registry client
            return {"success": False, "error": "No registry client available"}
            
        except Exception as e:
            logger.error(f"Failed to stage data via Data Manager: {e}")
            return {"success": False, "error": str(e)}
    
    async def _assess_dublin_core_quality(self, dublin_core: Dict[str, Any]) -> Dict[str, Any]:
        """Assess Dublin Core metadata quality according to ISO 15836, RFC 5013, ANSI/NISO Z39.85 with rigorous standards validation"""
        
        # Define actual Dublin Core 15 elements as per ISO 15836
        core_elements = ["title", "creator", "subject", "description", "publisher", 
                        "contributor", "date", "type", "format", "identifier", 
                        "source", "language", "relation", "coverage", "rights"]
        
        # Define valid DCMI Type Vocabulary as per ISO 15836
        dcmi_types = {
            "Collection", "Dataset", "Event", "Image", "InteractiveResource", 
            "MovingImage", "PhysicalObject", "Service", "Software", "Sound", 
            "StillImage", "Text"
        }
        
        # ISO 639-1 language codes (subset for validation)
        iso639_codes = {
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
            "eng", "spa", "fra", "deu", "ita", "por", "rus", "zho", "jpn", "kor", "ara", "hin"
        }
        
        populated = sum(1 for elem in core_elements if dublin_core.get(elem))
        completeness = populated / len(core_elements)
        
        # Rigorous format compliance checking
        accuracy = 1.0
        compliance_errors = []
        
        # Validate date format (ISO 8601) - RFC 5013 requirement
        if dublin_core.get("date"):
            try:
                # Support various ISO 8601 formats
                date_str = dublin_core["date"]
                if 'T' in date_str:
                    datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    datetime.strptime(date_str, '%Y-%m-%d')
            except (ValueError, TypeError):
                accuracy -= 0.15
                compliance_errors.append(f"Invalid date format: {dublin_core['date']} (must be ISO 8601)")
        
        # Validate language code (ISO 639-1/639-3) - ANSI/NISO Z39.85 requirement
        if dublin_core.get("language"):
            lang = dublin_core["language"].lower()
            if lang not in iso639_codes:
                accuracy -= 0.1
                compliance_errors.append(f"Invalid language code: {lang} (must be ISO 639)")
        
        # Validate DCMI Type Vocabulary - ISO 15836 requirement
        if dublin_core.get("type"):
            dc_type = dublin_core["type"]
            if dc_type not in dcmi_types:
                accuracy -= 0.1
                compliance_errors.append(f"Invalid type: {dc_type} (must be DCMI Type)")
        
        # Validate identifier format (should be URI or similar)
        if dublin_core.get("identifier"):
            identifier = dublin_core["identifier"]
            if not (identifier.startswith(('http://', 'https://', 'urn:', 'doi:', 'isbn:', 'issn:')) or 
                   identifier.count(':') >= 1):
                accuracy -= 0.05
                compliance_errors.append(f"Identifier should be URI-like: {identifier}")
        
        # Validate rights statement format
        if dublin_core.get("rights"):
            rights = dublin_core["rights"]
            if not (rights.startswith(('http://', 'https://')) or 
                   any(term in rights.lower() for term in ['copyright', 'license', 'public domain', 'creative commons'])):
                accuracy -= 0.05
                compliance_errors.append("Rights should be a standard license or copyright statement")
        
        # Enhanced consistency checks - RFC 5013 requirements
        consistency = 1.0
        consistency_errors = []
        
        # Title-Description semantic consistency
        if dublin_core.get("title") and dublin_core.get("description"):
            title_words = set(dublin_core["title"].lower().split())
            desc_words = set(dublin_core["description"].lower().split())
            overlap = len(title_words.intersection(desc_words))
            if overlap == 0 and len(title_words) > 1:
                consistency -= 0.15
                consistency_errors.append("Title and description share no common terms")
        
        # Creator-Publisher consistency (should not be identical unless same entity)
        if dublin_core.get("creator") and dublin_core.get("publisher"):
            if dublin_core["creator"] == dublin_core["publisher"]:
                consistency -= 0.05  # Minor issue, often valid
        
        # Subject-Description consistency
        if dublin_core.get("subject") and dublin_core.get("description"):
            subjects = dublin_core["subject"] if isinstance(dublin_core["subject"], list) else [dublin_core["subject"]]
            desc = dublin_core["description"].lower()
            subject_matches = sum(1 for subj in subjects if subj.lower() in desc)
            if subject_matches == 0:
                consistency -= 0.1
                consistency_errors.append("Subject terms not reflected in description")
        
        # Enhanced richness assessment - ANSI/NISO Z39.85 quality metrics
        richness = 0.0
        richness_factors = []
        
        # Multiple creators (collaborative work indication)
        if isinstance(dublin_core.get("creator"), list) and len(dublin_core["creator"]) > 1:
            richness += 0.2
            richness_factors.append("Multiple creators")
        
        # Rich subject indexing (3+ terms recommended)
        if isinstance(dublin_core.get("subject"), list) and len(dublin_core["subject"]) >= 3:
            richness += 0.2
            richness_factors.append("Rich subject indexing")
        
        # Contributors present (indicates collaboration)
        if isinstance(dublin_core.get("contributor"), list) and len(dublin_core["contributor"]) > 0:
            richness += 0.15
            richness_factors.append("Contributors specified")
        
        # Relations to other resources
        if dublin_core.get("relation"):
            richness += 0.15
            richness_factors.append("Resource relations")
        
        # Coverage information (temporal/spatial)
        if dublin_core.get("coverage"):
            richness += 0.1
            richness_factors.append("Coverage information")
        
        # Source attribution
        if dublin_core.get("source"):
            richness += 0.1
            richness_factors.append("Source attribution")
        
        # Format specification (technical metadata)
        if dublin_core.get("format"):
            richness += 0.1
            richness_factors.append("Format specification")
        
        # Mandatory elements check (Title, Creator, Date are often required)
        mandatory_score = 1.0
        mandatory_elements = ["title", "creator", "date"]
        missing_mandatory = [elem for elem in mandatory_elements if not dublin_core.get(elem)]
        if missing_mandatory:
            mandatory_score = 1.0 - (len(missing_mandatory) / len(mandatory_elements))
        
        # Calculate weighted overall score with mandatory elements priority
        overall_score = (
            completeness * 0.25 +      # Element population
            accuracy * 0.30 +          # Standards compliance (highest weight)
            consistency * 0.20 +       # Internal consistency
            richness * 0.15 +          # Metadata richness
            mandatory_score * 0.10     # Mandatory elements
        )
        
        # Rigorous standards compliance thresholds
        iso15836_compliant = (accuracy >= 0.9 and completeness >= 0.8 and mandatory_score == 1.0)
        rfc5013_compliant = (accuracy >= 0.85 and consistency >= 0.8 and overall_score >= 0.75)
        ansi_niso_compliant = (richness >= 0.5 and completeness >= 0.75 and overall_score >= 0.7)
        rfc5013_compliant = overall_score >= 0.75
        ansi_niso_compliant = overall_score >= 0.7
        
        quality_assessment = {
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "richness": richness,
            "overall_score": overall_score,
            "standards_compliance": {
                "iso15836_compliant": iso15836_compliant,
                "rfc5013_compliant": rfc5013_compliant,
                "ansi_niso_compliant": ansi_niso_compliant
            },
            "populated_elements": populated,
            "total_elements": len(core_elements),
            "recommendations": []
        }
        
        # Generate recommendations
        if completeness < 0.8:
            missing = [elem for elem in core_elements if not dublin_core.get(elem)]
            if missing:
                quality_assessment["recommendations"].append(
                    f"Consider adding missing elements: {', '.join(missing[:3])}"
                )
        
        if richness < 0.5:
            quality_assessment["recommendations"].append(
                "Enrich metadata with multiple creators, more subject keywords, and relations"
            )
        
        return quality_assessment
    
    async def _enhance_dublin_core_metadata(self, dublin_core: Dict[str, Any], 
                                          quality: Dict[str, Any], 
                                          data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Dublin Core metadata to meet quality threshold"""
        enhanced = dublin_core.copy()
        
        # Add missing critical elements
        if not enhanced.get("identifier"):
            enhanced["identifier"] = f"finsight-data-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        if not enhanced.get("rights"):
            enhanced["rights"] = "Internal Use Only - Proprietary Financial Data"
        
        if not enhanced.get("coverage"):
            enhanced["coverage"] = f"Financial Data Scope - {datetime.utcnow().year}"
        
        # Enhance subject keywords
        if isinstance(enhanced.get("subject"), list):
            # Add data-specific subjects
            for file_info in data_analysis.get("data_files", []):
                data_type = file_info.get("data_type", "")
                if data_type and data_type not in enhanced["subject"]:
                    enhanced["subject"].append(data_type)
            
            # Add domain subjects
            enhanced["subject"].extend(["financial-reporting", "enterprise-data", "standardization-required"])
            enhanced["subject"] = list(set(enhanced["subject"]))  # Remove duplicates
        
        # Enhance relations
        if not enhanced.get("relation") or len(enhanced["relation"]) < 2:
            enhanced["relation"] = [
                "com.finsight.cib:pipeline:financial_standardization",
                "com.finsight.cib:system:crd",
                "com.finsight.cib:process:data_extraction"
            ]
        
        # Enhance contributors
        if isinstance(enhanced.get("contributor"), list) and len(enhanced["contributor"]) < 3:
            enhanced["contributor"].extend([
                "Automated Extraction Process",
                "Data Quality Team",
                "Financial Data Governance"
            ])
            enhanced["contributor"] = list(set(enhanced["contributor"]))
        
        # Add format details
        if enhanced.get("format") == "text/csv":
            enhanced["format"] = "text/csv; charset=utf-8"
        
        return enhanced
    
    async def _generate_ord_descriptors(self, data_analysis: Dict[str, Any], cds_csn: Dict[str, Any], dublin_core_metadata: Dict[str, Any], staging_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate ORD descriptors for the data products with Dublin Core integration"""
        logger.info("=== DEBUG: Generating ORD Descriptors ===")
        logger.info(f"Data analysis keys: {list(data_analysis.keys()) if isinstance(data_analysis, dict) else 'Not a dict'}")
        logger.info(f"CDS CSN keys: {list(cds_csn.keys()) if isinstance(cds_csn, dict) else 'Not a dict'}")
        logger.info(f"Dublin Core metadata keys: {list(dublin_core_metadata.keys()) if isinstance(dublin_core_metadata, dict) else 'Not a dict'}")
        logger.info(f"Staging info: {'Present' if staging_info else 'None'}")
        
        data_products = []
        entity_types = []
        
        for file_info in data_analysis["data_files"]:
            data_type = file_info["data_type"]
            
            # Get staging information for this data type
            data_staging_info = staging_info.get(data_type, {}) if staging_info else {}
            access_strategy = data_staging_info.get("access_strategy", {
                "type": "file",
                "path": file_info["path"]
            })
            
            # Create data product descriptor with Dublin Core elements
            data_product = {
                "ordId": f"com.finsight.cib:dataProduct:crd_{data_type}_data",
                "title": f"CRD {data_type.capitalize()} Data",
                "shortDescription": f"{data_type.capitalize()} data - {file_info['records']} records",
                "description": f"Financial {data_type} data extracted from CRD system",
                "version": "1.0.0",
                "visibility": "internal",
                "tags": ["crd", "financial", data_type, "raw-data"],
                "labels": {
                    "source": "crd_extraction",
                    "format": "csv" if access_strategy["type"] == "file" else "database",
                    "records": str(file_info['records']),
                    "columns": str(len(file_info['columns'])),
                    "integrity_hash": file_info.get('integrity', {}).get('dataset_hash', ''),
                    "row_count": str(file_info.get('integrity', {}).get('row_count', 0)),
                    "staging_method": data_staging_info.get("staging_method", "file")
                },
                "accessStrategies": [access_strategy],
                "dublinCore": {
                    "title": f"CRD {data_type.capitalize()} Data",
                    "creator": dublin_core_metadata.get("creator", ["FinSight CIB"]),
                    "subject": [data_type, "financial-data", "crd"],
                    "description": f"Financial {data_type} data extracted from CRD system with {file_info['records']} records",
                    "type": dublin_core_metadata.get("type", "Dataset"),
                    "format": dublin_core_metadata.get("format", "text/csv")
                },
                "integrity": file_info.get("integrity", {}),
                "stagingInfo": data_staging_info if data_staging_info else None
            }
            data_products.append(data_product)
            
            # Create entity type descriptor
            entity_type = {
                "ordId": f"com.finsight.cib:entityType:{data_type.capitalize()}",
                "title": f"{data_type.capitalize()} Entity",
                "shortDescription": f"{data_type.capitalize()} entity type",
                "description": f"Entity type for {data_type} data",
                "version": "1.0.0",
                "visibility": "internal",
                "tags": ["entity", data_type, "cds"]
            }
            entity_types.append(entity_type)
        
        # Create complete ORD document with enhanced Dublin Core metadata
        ord_document = {
            "openResourceDiscovery": "1.5.0",
            "description": "CRD Financial Data Products with enhanced Dublin Core metadata",
            "dublinCore": dublin_core_metadata,
            "dataProducts": data_products,
            "entityTypes": entity_types,
            "cdsSchema": cds_csn
        }
        
        # Log the final ORD document structure
        logger.info("=== DEBUG: Final ORD Document Structure ===")
        logger.info(f"Number of data products: {len(data_products)}")
        logger.info(f"Number of entity types: {len(entity_types)}")
        if data_products:
            logger.info(f"First data product keys: {list(data_products[0].keys())}")
        if entity_types:
            logger.info(f"First entity type keys: {list(entity_types[0].keys())}")
        
        # Check for any None values that might cause issues
        for key, value in ord_document.items():
            if value is None:
                logger.warning(f"ORD document has None value for key: {key}")
        
        return ord_document
    
    async def _register_in_catalog(self, ord_descriptors: Dict[str, Any]) -> Dict[str, Any]:
        """Register data products in ORD Registry via Catalog Manager A2A"""
        try:
            logger.info(f"=== Agent 0 ORD Registration via Catalog Manager ===")
            logger.info(f"ORD document size: {len(str(ord_descriptors))} characters")
            
            # Create A2A message for ORD registration
            registration_request = {
                "operation": "register",
                "ord_document": ord_descriptors,
                "enhancement_type": "metadata_enrichment",
                "ai_powered": True,
                "context": {
                    "source_agent": self.agent_id,
                    "registered_by": "data_product_agent",
                    "purpose": "data_product_registration",
                    "tags": ["automated", "pipeline"],
                    "labels": {
                        "agent": "data_product_registration",
                        "pipeline": "financial_standardization"
                    }
                }
            }
            
            message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="text",
                        text=json.dumps(registration_request)
                    )
                ],
                taskId=str(uuid4()),
                contextId=str(uuid4())
            )
            
            # Send message to Catalog Manager via Agent Manager discovery
            discovery_result = await self._discover_agents_via_manager(skills=["ord-management"])
            if discovery_result.get("success", False):
                # Get Catalog Manager endpoint
                agents = discovery_result.get("agents", [])
                catalog_manager = next((agent for agent in agents if "catalog" in agent.get("id", "").lower()), None)
                
                if catalog_manager:
                    catalog_manager_url = catalog_manager.get("url")
                    if not catalog_manager_url:
                        logger.error("Catalog Manager agent found but no URL provided")
                        return {"success": False, "error": "Catalog Manager URL not available"}
                    
                    # Send A2A message
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{catalog_manager_url}/a2a/catalog_manager/v1/message",
                            json={
                                "message": message.model_dump(),
                                "contextId": message.contextId
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("status") == "success":
                                logger.info(f"✅ ORD registration successful via Catalog Manager")
                                return result.get("result", {})
                            else:
                                error_msg = result.get("message", "Unknown error")
                                logger.error(f"❌ Catalog Manager registration failed: {error_msg}")
                                raise Exception(f"Registration failed: {error_msg}")
                        else:
                            logger.error(f"❌ Catalog Manager HTTP error: {response.status_code} - {response.text}")
                            raise Exception(f"HTTP {response.status_code}: {response.text}")
                else:
                    logger.error("❌ Catalog Manager agent not found in registry")
                    raise Exception("Catalog Manager agent not available")
            else:
                logger.error("❌ No registry client available")
                raise Exception("Registry client not available")
                    
        except Exception as e:
            logger.error(f"Exception during ORD registration via Catalog Manager: {type(e).__name__}: {str(e)}")
            raise
    
    async def _trigger_downstream_agent(self, catalog_registration: Dict[str, Any], context_id: str, dublin_core_quality: Dict[str, Any], workflow_context=None) -> Dict[str, Any]:
        """Trigger the standardization agent with catalog reference and Dublin Core context"""
        try:
            # Try to discover Agent 1 dynamically via Agent Manager
            downstream_agent_url = None
            discovered_agent = None
            
            try:
                # Search for agents with standardization skills via Agent Manager
                logger.info("Searching for agents with standardization capabilities...")
                discovery_result = await self._discover_agents_via_manager(skills=["batch-standardization"])
                
                if discovery_result.get("success", False) and discovery_result.get("agents"):
                    # Pick the first available agent
                    discovered_agent = discovery_result["agents"][0]
                    downstream_agent_url = discovered_agent.get("url")
                    logger.info(f"✅ Discovered standardization agent: {discovered_agent.get('agent_id')} at {downstream_agent_url}")
                else:
                    logger.warning("No healthy standardization agents found via dynamic discovery")
                    
            except Exception as e:
                logger.warning(f"Dynamic discovery failed: {str(e)}")
            
            # Fall back to configured URL if discovery fails
            if not downstream_agent_url:
                if self.downstream_agent_url:
                    downstream_agent_url = self.downstream_agent_url
                    logger.info(f"Using configured downstream agent URL: {downstream_agent_url}")
                else:
                    raise Exception("No downstream agent URL available (neither discovered nor configured)")
            
            # Create A2A message for standardization agent with ORD reference
            trigger_message = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Standardize financial data products registered in ORD catalog"
                        },
                        {
                            "kind": "data",
                            "data": {
                                "task_instruction": {
                                    "action": "standardize_data_products",
                                    "description": "Retrieve data products from ORD registry and apply standardization"
                                },
                                "ord_reference": {
                                    "registry_url": catalog_registration.get("registry_url"),
                                    "registration_id": catalog_registration.get("registration_id"),
                                    "resource_type": "dataProduct",
                                    "query_params": {
                                        "tags": ["crd", "raw-data"],
                                        "registered_by": "data_product_agent"
                                    }
                                },
                                "processing_requirements": {
                                    "data_types": ["account", "location", "product", "book", "measure"],
                                    "standardization_level": "L4",
                                    "output_storage": {
                                        "method": "ord_registry",
                                        "format": "cds_csn",
                                        "location": "/data/interim/1/dataStandardization"
                                    }
                                },
                                "workflow_context": workflow_context_manager.serialize_for_message(
                                    workflow_context.workflow_id
                                ) if workflow_context else {
                                    "pipeline_id": "financial_processing_pipeline",
                                    "stage": "standardization",
                                    "initiated_by": "data_product_agent",
                                    "upstream_task_id": context_id,
                                    "compliance": {
                                        "a2a_version": "0.2.9",
                                        "ord_version": "1.5.0"
                                    }
                                },
                                "dublin_core_context": {
                                    "quality_score": dublin_core_quality.get("overall_score", 0.0),
                                    "standards_compliant": dublin_core_quality.get("standards_compliance", {}).get("iso15836_compliant", False),
                                    "completeness": dublin_core_quality.get("completeness", 0.0),
                                    "populated_elements": dublin_core_quality.get("populated_elements", 0),
                                    "total_elements": dublin_core_quality.get("total_elements", 15)
                                },
                                "discovery_metadata": {
                                    "discovered_via": "a2a_registry" if discovered_agent else "config",
                                    "agent_id": discovered_agent.get("agent_id") if discovered_agent else None,
                                    "agent_name": discovered_agent.get("name") if discovered_agent else None
                                }
                            }
                        }
                    ]
                },
                "contextId": f"pipeline_{context_id}"
            }
            
            # Sign the A2A message with Agent 0's private key for trusted communication
            signed_message = sign_a2a_message(self.agent_id, trigger_message)
            logger.info(f"✅ A2A message signed for trusted communication to Agent 1")
            
            # If we discovered an agent, use its specific endpoint
            # Otherwise use the configured URL
            if discovered_agent:
                # Agent 1 expects messages at /a2a/v1/messages
                endpoint = f"{downstream_agent_url.rstrip('/')}/a2a/v1/messages"
            else:
                # Fallback to configured URL with /process endpoint
                endpoint = f"{downstream_agent_url.rstrip('/')}/process"
            
            logger.info(f"Sending message to downstream agent at: {endpoint}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json=signed_message,
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9",
                        "X-Trust-Contract": get_trust_contract().contract_id,
                        "X-Agent-Identity": self.agent_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "agent": discovered_agent.get("agent_id") if discovered_agent else "financial-data-standardization-agent",
                        "agent_url": downstream_agent_url,
                        "discovered": discovered_agent is not None,
                        "task_id": result.get("taskId"),
                        "status": "triggered",
                        "message": trigger_message,
                        "signed_message": signed_message,
                        "trust_verification": "message_signed",
                        "contract_id": get_trust_contract().contract_id
                    }
                else:
                    raise Exception(f"Failed to trigger downstream agent: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error triggering downstream agent: {str(e)}")
            # Don't fail the whole process if downstream trigger fails
            return {
                "agent": "financial-data-standardization-agent",
                "status": "trigger_failed",
                "error": str(e)
            }
    
    async def _update_status(self, task_id: str, state: TaskState, message: str = None, error: Dict = None):
        """Update task status"""
        status = TaskStatus(state=state, error=error)
        
        if message:
            status.message = A2AMessage(
                role=MessageRole.AGENT,
                parts=[MessagePart(kind="text", text=message)],
                taskId=task_id,
                contextId=self.tasks[task_id]["contextId"]
            )
        
        self.tasks[task_id]["status"] = status
        
        # Persist state after status update
        asyncio.create_task(self._persist_state())
        self.tasks[task_id]["events"].append({
            "type": "status-update",
            "timestamp": datetime.utcnow().isoformat(),
            "status": status.dict()
        })
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current task status"""
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.tasks[task_id]
        return {
            "taskId": task_id,
            "contextId": task["contextId"],
            "status": task["status"].dict(),
            "artifacts": [a.dict() for a in task["artifacts"]]
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
    
    async def get_advisor_stats(self) -> Dict[str, Any]:
        """Get AI advisor statistics"""
        return self.ai_advisor.get_advisor_stats()
    
    async def _load_persisted_state(self):
        """Load persisted tasks and cancelled tasks from storage"""
        try:
            # Load tasks
            tasks_file = os.path.join(self._storage_path, "tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                
                # Deserialize TaskStatus objects
                for task_id, task_info in tasks_data.items():
                    if 'status' in task_info and isinstance(task_info['status'], dict):
                        # Handle datetime deserialization
                        if 'timestamp' in task_info['status']:
                            task_info['status']['timestamp'] = task_info['status']['timestamp']
                        task_info['status'] = TaskStatus(**task_info['status'])
                    
                    # Deserialize artifacts
                    if 'artifacts' in task_info:
                        artifacts = []
                        for artifact_data in task_info['artifacts']:
                            artifacts.append(TaskArtifact(**artifact_data))
                        task_info['artifacts'] = artifacts
                    
                    self.tasks[task_id] = task_info
                
                logger.info(f"✅ Loaded {len(self.tasks)} tasks from storage")
            
            # Load cancelled tasks
            cancelled_file = os.path.join(self._storage_path, "cancelled_tasks.json")
            if os.path.exists(cancelled_file):
                with open(cancelled_file, 'r') as f:
                    self.cancelled_tasks = set(json.load(f))
                logger.info(f"✅ Loaded {len(self.cancelled_tasks)} cancelled tasks from storage")
                
        except Exception as e:
            logger.error(f"❌ Failed to load persisted state: {e}")
    
    async def _persist_state(self):
        """Persist current tasks and cancelled tasks to storage"""
        try:
            # Serialize tasks
            tasks_data = {}
            for task_id, task_info in self.tasks.items():
                # Create a serializable copy
                serializable_task = {}
                for key, value in task_info.items():
                    if key == 'status' and hasattr(value, 'model_dump'):
                        serializable_task[key] = value.model_dump()
                    elif key == 'artifacts':
                        serializable_task[key] = [artifact.model_dump() for artifact in value]
                    else:
                        serializable_task[key] = value
                tasks_data[task_id] = serializable_task
            
            # Save tasks
            tasks_file = os.path.join(self._storage_path, "tasks.json")
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Save cancelled tasks
            cancelled_file = os.path.join(self._storage_path, "cancelled_tasks.json")
            with open(cancelled_file, 'w') as f:
                json.dump(list(self.cancelled_tasks), f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to persist state: {e}")
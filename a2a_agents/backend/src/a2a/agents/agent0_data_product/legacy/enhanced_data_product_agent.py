"""
Enhanced Data Product Registration Agent (Agent 0) with AI Advisor and Smart Contract Delegation
"""

import asyncio
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
import logging
import httpx
import hashlib

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .data_standardization_agent import (
    TaskState, MessageRole, A2AMessage, MessagePart,
    TaskStatus, TaskArtifact, AgentCard
)
from app.a2a.core.workflow_context import workflow_context_manager, DataArtifact
from app.a2a.core.workflow_monitor import workflow_monitor
from app.clients.supabase_client import create_supabase_client
from app.clients.hana_client import create_hana_client
from ..security.smart_contract_trust import initialize_agent_trust, sign_a2a_message, get_trust_contract
from ..security.delegation_contracts import get_delegation_contract, DelegationAction, can_agent_delegate, record_delegation_usage
from app.a2a_registry.client import get_registry_client
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor

logger = logging.getLogger(__name__)


class EnhancedDataProductRegistrationAgent:
    """Enhanced Agent 0: Data Product Registration Agent with AI Advisor and Smart Contract Delegation"""
    
    def __init__(self, base_url: str, ord_registry_url: str, downstream_agent_url: str = None):
        self.base_url = base_url
        self.ord_registry_url = ord_registry_url
        self.downstream_agent_url = downstream_agent_url
        self.registry_client = None
        
        # Initialize smart contract trust identity
        self.agent_id = "data_product_agent_0"
        self.agent_identity = initialize_agent_trust(
            self.agent_id,
            "DataProductRegistrationAgent"
        )
        logger.info(f"✅ Enhanced Agent 0 trust identity initialized: {self.agent_id}")
        
        # Agent capabilities
        self.capabilities = {
            "streaming": True,
            "pushNotifications": True,
            "stateTransitionHistory": True,
            "batchProcessing": True,
            "metadataExtraction": True,
            "dublinCoreCompliance": True,
            "smartContractDelegation": True,
            "aiAdvisor": True
        }
        
        self.agent_card = AgentCard(
            name="Enhanced Data Product Registration Agent",
            description="Processes raw data into CDS schema with ORD descriptors, enhanced with AI advisor and smart contract delegation",
            url=base_url,
            version="2.0.0",
            protocolVersion="0.2.9",
            provider={
                "organization": "FinSight CIB",
                "url": "https://finsight-cib.com"
            },
            capabilities=self.capabilities,
            defaultInputModes=["text/plain", "text/csv", "application/json", "application/xml"],
            defaultOutputModes=["application/json", "application/cds", "application/ord+json"],
            skills=[
                {
                    "id": "dublin-core-extraction",
                    "name": "Dublin Core Metadata Extraction",
                    "description": "Extract and generate Dublin Core metadata from raw data",
                    "tags": ["dublin-core", "metadata", "standards"],
                    "inputModes": ["text/csv", "application/json"],
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
        
        # Initialize AI Advisor
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name="Data Product Registration Agent",
            agent_capabilities=self.capabilities
        )
        
        # Add knowledge to AI advisor
        self._initialize_advisor_knowledge()
        
        # Task management
        self.tasks = {}
        self.cancelled_tasks = set()
        
        logger.info("✅ Enhanced Agent 0 initialized with AI advisor and delegation capabilities")
    
    def _initialize_advisor_knowledge(self):
        """Initialize AI advisor with agent-specific knowledge"""
        # Add FAQs
        self.ai_advisor.add_faq_item(
            "What does Agent 0 do?",
            "I am the Data Product Registration Agent. I process raw financial data, extract Dublin Core metadata, create CDS schemas, generate ORD descriptors, and register data products in the catalog. I can delegate data storage to the Data Manager and metadata registration to the Catalog Manager."
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
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card"""
        return self.agent_card.dict()
    
    async def process_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Process incoming A2A message with AI advisor support"""
        task_id = str(uuid4())
        
        # Check if this is an AI advisor request
        if self._is_advisor_request(message):
            return await self._handle_advisor_request(message, context_id)
        
        # Extract workflow context from message if available
        workflow_context = None
        workflow_id = None
        
        for part in message.parts:
            if part.kind == "data" and "workflow_context" in part.data:
                wf_data = part.data["workflow_context"]
                workflow_id = wf_data.get("workflow_id")
                workflow_context = workflow_context_manager.get_context(workflow_id)
                break
        
        # Initialize task
        self.tasks[task_id] = {
            "taskId": task_id,
            "contextId": context_id,
            "workflowId": workflow_id,
            "status": TaskStatus(state=TaskState.PENDING),
            "artifacts": [],
            "events": []
        }
        
        # Update AI advisor with activity
        self.ai_advisor.log_activity({
            "action": "message_received",
            "task_id": task_id,
            "context_id": context_id,
            "message_type": "data_registration_request"
        })
        
        # Start processing in background
        asyncio.create_task(self._execute_enhanced_registration(
            task_id, message, context_id, workflow_context
        ))
        
        return {
            "taskId": task_id,
            "contextId": context_id,
            "workflowId": workflow_id,
            "status": self.tasks[task_id]["status"].dict(),
            "ai_advisor_available": True
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
    
    async def _execute_enhanced_registration(
        self, 
        task_id: str, 
        message: A2AMessage, 
        context_id: str,
        workflow_context=None
    ):
        """Execute enhanced data product registration with delegation"""
        try:
            await self._update_status(task_id, TaskState.WORKING, "Starting enhanced registration with delegation...")
            
            # Extract data location from message
            data_location = self._extract_data_location(message)
            
            # Step 1: Analyze raw data
            await self._update_status(task_id, TaskState.WORKING, "Analyzing data structure...")
            data_analysis = await self._analyze_raw_data(data_location)
            
            # Update AI advisor with current status
            self.ai_advisor.update_agent_status({
                "current_task": task_id,
                "processing_stage": "data_analysis",
                "data_files_found": len(data_analysis.get("data_files", [])),
                "total_records": data_analysis.get("total_records", 0)
            })
            
            # Step 2: Delegate data storage to Data Manager
            await self._update_status(task_id, TaskState.WORKING, "Delegating data storage to Data Manager...")
            storage_result = await self._delegate_data_storage(data_analysis, context_id)
            
            # Step 3: Extract Dublin Core metadata
            await self._update_status(task_id, TaskState.WORKING, "Extracting Dublin Core metadata...")
            dublin_core_metadata = await self._extract_dublin_core_metadata(data_analysis)
            
            # Step 4: Generate CDS CSN
            await self._update_status(task_id, TaskState.WORKING, "Generating CDS schema...")
            cds_csn = await self._generate_cds_csn(data_analysis)
            
            # Step 5: Delegate ORD registration to Catalog Manager
            await self._update_status(task_id, TaskState.WORKING, "Delegating ORD registration to Catalog Manager...")
            ord_registration = await self._delegate_ord_registration(
                data_analysis, cds_csn, dublin_core_metadata, context_id
            )
            
            # Step 6: Trigger downstream agent (Agent 1)
            await self._update_status(task_id, TaskState.WORKING, "Triggering standardization agent...")
            downstream_trigger = await self._trigger_downstream_agent(
                ord_registration, context_id, workflow_context
            )
            
            # Create final artifact
            artifact = TaskArtifact(
                name="Enhanced Registration Results",
                description="Complete data product registration with smart contract delegation",
                parts=[{
                    "kind": "data",
                    "data": {
                        "data_analysis": data_analysis,
                        "storage_delegation": storage_result,
                        "dublin_core_metadata": dublin_core_metadata,
                        "cds_csn": cds_csn,
                        "ord_registration": ord_registration,
                        "downstream_trigger": downstream_trigger,
                        "delegation_summary": {
                            "data_manager_used": storage_result.get("success", False),
                            "catalog_manager_used": ord_registration.get("success", False),
                            "agent1_triggered": downstream_trigger.get("status") == "triggered"
                        }
                    }
                }]
            )
            
            self.tasks[task_id]["artifacts"].append(artifact)
            
            # Update AI advisor with completion
            self.ai_advisor.log_activity({
                "action": "task_completed",
                "task_id": task_id,
                "delegations_used": ["data_manager", "catalog_manager"],
                "downstream_triggered": True
            })
            
            await self._update_status(
                task_id,
                TaskState.COMPLETED,
                "Enhanced registration completed successfully with smart contract delegation"
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ Enhanced registration error: {str(e)}")
            
            # Update AI advisor with error
            self.ai_advisor.log_activity({
                "action": "task_failed",
                "task_id": task_id,
                "error": str(e)
            })
            
            await self._update_status(
                task_id,
                TaskState.FAILED,
                error={
                    "code": "ENHANCED_REGISTRATION_ERROR",
                    "message": str(e),
                    "traceback": error_details
                }
            )
    
    async def _delegate_data_storage(self, data_analysis: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Delegate data storage operations to Data Manager Agent"""
        try:
            # Check delegation authorization
            if not can_agent_delegate(
                self.agent_id,
                "data_manager_agent",
                DelegationAction.DATA_STORAGE,
                {"trust_score": 0.8}
            ):
                raise Exception("Delegation to Data Manager not authorized")
            
            # Create delegation message for Data Manager
            delegation_message = A2AMessage(
                role=MessageRole.USER,
                contextId=context_id,
                parts=[{
                    "kind": "data",
                    "data": {
                        "operation": "create",
                        "storage_type": "dual",
                        "service_level": "silver",
                        "delegation_request": {
                            "delegator": self.agent_id,
                            "action": DelegationAction.DATA_STORAGE.value,
                            "data_analysis": data_analysis
                        }
                    }
                }]
            )
            
            # Sign message
            signed_message = sign_a2a_message(self.agent_id, delegation_message.dict())
            
            # Send to Data Manager
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8003/a2a/v1/messages",
                    json=signed_message,
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9",
                        "X-Delegation-Contract": get_delegation_contract().contract_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Record successful delegation
                    record_delegation_usage(
                        self.agent_id,
                        "data_manager_agent",
                        DelegationAction.DATA_STORAGE,
                        True,
                        {"context_id": context_id}
                    )
                    
                    return {
                        "success": True,
                        "data_manager_task_id": result.get("taskId"),
                        "delegation_used": True
                    }
                else:
                    raise Exception(f"Data Manager delegation failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"❌ Data storage delegation failed: {e}")
            
            # Record failed delegation
            record_delegation_usage(
                self.agent_id,
                "data_manager_agent", 
                DelegationAction.DATA_STORAGE,
                False,
                {"error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True
            }
    
    async def _delegate_ord_registration(
        self, 
        data_analysis: Dict[str, Any], 
        cds_csn: Dict[str, Any],
        dublin_core_metadata: Dict[str, Any],
        context_id: str
    ) -> Dict[str, Any]:
        """Delegate ORD registration to Catalog Manager Agent"""
        try:
            # Check delegation authorization
            if not can_agent_delegate(
                self.agent_id,
                "catalog_manager_agent",
                DelegationAction.METADATA_REGISTRATION,
                {"trust_score": 0.8}
            ):
                raise Exception("Delegation to Catalog Manager not authorized")
            
            # Create ORD document for registration
            ord_document = await self._generate_ord_descriptors(
                data_analysis, cds_csn, dublin_core_metadata
            )
            
            # Create delegation message for Catalog Manager
            delegation_message = A2AMessage(
                role=MessageRole.USER,
                contextId=context_id,
                parts=[{
                    "kind": "data",
                    "data": {
                        "operation": "register",
                        "ord_document": ord_document,
                        "ai_powered": True,
                        "delegation_request": {
                            "delegator": self.agent_id,
                            "action": DelegationAction.METADATA_REGISTRATION.value
                        }
                    }
                }]
            )
            
            # Sign message
            signed_message = sign_a2a_message(self.agent_id, delegation_message.dict())
            
            # Send to Catalog Manager
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8005/a2a/catalog_manager/v1/message",
                    json=signed_message,
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9",
                        "X-Delegation-Contract": get_delegation_contract().contract_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Record successful delegation
                    record_delegation_usage(
                        self.agent_id,
                        "catalog_manager_agent",
                        DelegationAction.METADATA_REGISTRATION,
                        True,
                        {"context_id": context_id}
                    )
                    
                    return {
                        "success": True,
                        "catalog_manager_task_id": result.get("taskId"),
                        "delegation_used": True,
                        "ord_document": ord_document
                    }
                else:
                    raise Exception(f"Catalog Manager delegation failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"❌ ORD registration delegation failed: {e}")
            
            # Record failed delegation
            record_delegation_usage(
                self.agent_id,
                "catalog_manager_agent",
                DelegationAction.METADATA_REGISTRATION,
                False,
                {"error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True
            }
    
    # ... (include other existing methods from original agent: _extract_data_location, _analyze_raw_data, etc.)
    
    def _extract_data_location(self, message: A2AMessage) -> str:
        """Extract data location from message"""
        for part in message.parts:
            if part.kind == "data":
                if "data_location" in part.data:
                    return part.data["data_location"]
                elif "processing_instructions" in part.data:
                    instructions = part.data.get("processing_instructions", {})
                    if isinstance(instructions, dict):
                        return "/Users/apple/projects/finsight_cib/data/raw"
        
        return "/Users/apple/projects/finsight_cib/data/raw"
    
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
                    data_type = filename.replace('CRD_Extraction_v1_', '').replace('_sorted.csv', '')
                    data_records = df.to_dict('records')
                    
                    file_info = {
                        "filename": filename,
                        "path": file_path,
                        "data_type": data_type,
                        "records": len(df),
                        "columns": list(df.columns),
                        "sample_data": df.head(3).to_dict('records')
                    }
                    
                    analysis["data_files"].append(file_info)
                    analysis["total_records"] += len(df)
                    analysis["data_types"].append(data_type)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {filename}: {str(e)}")
        
        return analysis
    
    async def _extract_dublin_core_metadata(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dublin Core metadata from data analysis"""
        all_data_types = data_analysis.get("data_types", [])
        total_records = data_analysis.get("total_records", 0)
        file_count = len(data_analysis.get("data_files", []))
        
        dublin_core = {
            "title": f"CRD Financial Data Products - {datetime.utcnow().strftime('%B %Y')}",
            "creator": ["FinSight CIB", "Enhanced Data Product Registration Agent"],
            "subject": ["financial-data", "crd-extraction", "enhanced-processing"] + all_data_types,
            "description": f"Enhanced financial data collection with {total_records:,} records across {file_count} data types using smart contract delegation",
            "publisher": "FinSight CIB Data Platform",
            "date": datetime.utcnow().isoformat(),
            "type": "Dataset",
            "format": "text/csv",
            "identifier": f"enhanced-crd-data-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "rights": "Internal Use Only - FinSight CIB Proprietary Data"
        }
        
        return dublin_core
    
    async def _generate_cds_csn(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CDS Core Schema Notation"""
        definitions = {}
        
        for file_info in data_analysis["data_files"]:
            data_type = file_info["data_type"]
            columns = file_info["columns"]
            
            entity_name = f"{data_type.capitalize()}Entity"
            elements = {}
            
            for col in columns:
                elements[col.replace(" ", "_").replace("(", "").replace(")", "")] = {
                    "type": "cds.String",
                    "length": 255
                }
            
            definitions[entity_name] = {
                "kind": "entity",
                "elements": elements
            }
        
        return {
            "definitions": definitions,
            "meta": {
                "creator": "EnhancedDataProductRegistrationAgent",
                "flavor": "enhanced",
                "namespace": "com.finsight.cib.enhanced"
            },
            "$version": "2.0"
        }
    
    async def _generate_ord_descriptors(
        self, 
        data_analysis: Dict[str, Any], 
        cds_csn: Dict[str, Any], 
        dublin_core_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ORD descriptors for the data products"""
        data_products = []
        
        for file_info in data_analysis["data_files"]:
            data_type = file_info["data_type"]
            
            data_product = {
                "ordId": f"com.finsight.cib:dataProduct:enhanced_{data_type}_data",
                "title": f"Enhanced CRD {data_type.capitalize()} Data",
                "description": f"Enhanced financial {data_type} data with smart contract delegation",
                "version": "2.0.0",
                "visibility": "internal",
                "tags": ["enhanced", "crd", "financial", data_type, "delegated"],
                "accessStrategies": [{
                    "type": "file",
                    "path": file_info["path"]
                }],
                "dublinCore": dublin_core_metadata
            }
            data_products.append(data_product)
        
        return {
            "openResourceDiscovery": "1.5.0",
            "description": "Enhanced CRD Financial Data Products with Smart Contract Delegation",
            "dublinCore": dublin_core_metadata,
            "dataProducts": data_products,
            "cdsSchema": cds_csn
        }
    
    async def _trigger_downstream_agent(
        self, 
        ord_registration: Dict[str, Any], 
        context_id: str,
        workflow_context=None
    ) -> Dict[str, Any]:
        """Trigger Agent 1 with enhanced messaging"""
        try:
            # Create enhanced trigger message
            trigger_message = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Process enhanced data products with smart contract delegation"
                        },
                        {
                            "kind": "data",
                            "data": {
                                "enhanced_processing": True,
                                "delegation_enabled": True,
                                "upstream_agent": self.agent_id,
                                "ord_registration": ord_registration
                            }
                        }
                    ]
                },
                "contextId": context_id
            }
            
            # Sign and send to Agent 1
            signed_message = sign_a2a_message(self.agent_id, trigger_message)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/a2a/v1/messages",
                    json=signed_message,
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "triggered",
                        "agent1_task_id": result.get("taskId"),
                        "enhanced_processing": True
                    }
                else:
                    return {
                        "status": "trigger_failed",
                        "error": response.text
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error triggering downstream agent: {e}")
            return {
                "status": "trigger_failed",
                "error": str(e)
            }
    
    async def _update_status(self, task_id: str, state: TaskState, message: str = None, error: Dict = None):
        """Update task status"""
        status = TaskStatus(state=state, error=error)
        
        if message:
            status.message = A2AMessage(
                role=MessageRole.AGENT,
                parts=[{"kind": "text", "text": message}],
                taskId=task_id,
                contextId=self.tasks[task_id]["contextId"]
            )
        
        self.tasks[task_id]["status"] = status
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
            "artifacts": [a.dict() for a in task["artifacts"]],
            "ai_advisor_available": True
        }
    
    async def get_advisor_stats(self) -> Dict[str, Any]:
        """Get AI advisor statistics"""
        return self.ai_advisor.get_advisor_stats()
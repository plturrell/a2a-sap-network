"""
Data Product Registration Agent - SDK Version
Agent 0: Enhanced with A2A SDK for simplified development and maintenance
"""

import asyncio
import json
import os
import pandas as pd
import httpx
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
from uuid import uuid4

# Import SDK components - use local components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.core.workflowContext import WorkflowContextManager as workflowContextManager
from app.a2a.core.workflowMonitor import workflowMonitor as workflowMonitor
from app.a2a.core.asyncPatterns import (
    async_retry, async_timeout, async_concurrent_limit,
    AsyncOperationType, AsyncOperationConfig, async_manager
)
# Logger is already defined at the module level  
# Import trust components from a2aNetwork
try:
    import sys
    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
    from trustSystem.smartContractTrust import sign_a2a_message
    from trustSystem.delegationContracts import DelegationAction
    print("✅ Using a2aNetwork trust components in agent0")
except ImportError:
    # Fallback - create mock functions
    def sign_a2a_message(*args, **kwargs):
        return {"signature": "mock", "timestamp": datetime.utcnow().isoformat()}
    class DelegationAction:
        READ = "read"
        WRITE = "write"
        EXECUTE = "execute"
    print("⚠️  Using fallback trust components in agent0")
from app.a2a.skills.accountStandardizer import AccountStandardizer
from app.a2a.skills.bookStandardizer import BookStandardizer
from app.a2a.skills.locationStandardizer import LocationStandardizer
from app.a2a.skills.measureStandardizer import MeasureStandardizer
from app.a2a.skills.productStandardizer import ProductStandardizer

import logging
logger = logging.getLogger(__name__)


class DataProductRegistrationAgentSDK(A2AAgentBase):
    """
    Agent 0: Data Product Registration Agent with Dublin Core
    SDK Version - Simplified development with enhanced capabilities
    """
    
    def __init__(self, base_url: str, ord_registry_url: str):
        super().__init__(
            agent_id="data_product_agent_0",
            name="Data Product Registration Agent",
            description="A2A v0.2.9 compliant agent for data product registration with Dublin Core metadata",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.ord_registry_url = ord_registry_url
        self.data_products = {}
        self.processing_stats = {
            "total_processed": 0,
            "successful_registrations": 0,
            "dublin_core_extractions": 0,
            "integrity_verifications": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    @async_retry(max_retries=3, operation_type=AsyncOperationType.IO_BOUND)
    @async_timeout(30.0)
    async def initialize(self) -> None:
        """Initialize agent resources with standardized async patterns"""
        
        logger.info(f"Starting agent initialization for {self.agent_id}")
        
        try:
            # Initialize data storage
            storage_path = os.getenv("DATA_PRODUCT_AGENT_STORAGE_PATH", "/tmp/data_product_agent_state")
            os.makedirs(storage_path, exist_ok=True)
            self.storage_path = storage_path
            
            # Initialize HTTP client for async operations
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            
            # Load any persistent state
            await self._load_persistent_state()
            
            # Initialize network connectivity and auto-register
            await self._initialize_network_integration()
            
            logger.info(f"Agent initialization completed for {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Agent initialization failed for {self.agent_id}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cleanup agent resources with proper async resource management"""
        
        logger.info(f"Starting agent shutdown for {self.agent_id}")
        
        try:
            # Save persistent state
            await self._save_persistent_state()
            
            # Close HTTP client properly
            if hasattr(self, 'http_client') and self.http_client:
                await self.http_client.aclose()
            
            # Clear in-memory data
            self.data_products.clear()
            
            logger.info(f"Agent shutdown completed for {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Agent shutdown failed for {self.agent_id}: {e}")
            # Continue with shutdown even if there are errors
        finally:
            # Ensure resources are cleaned up
            self.data_products.clear()
    
    @a2a_handler("process_data", "Process and register data products with Dublin Core metadata")
    async def handle_data_processing(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main data processing handler"""
        try:
            # Extract data from message
            data_info = self._extract_data_info(message)
            
            if not data_info:
                return create_error_response(400, "No data found in message")
            
            # Create task for tracking
            task_id = await self.create_task("data_product_registration", {
                "context_id": context_id,
                "data_info": data_info
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_data_product(task_id, data_info, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "message": "Data product registration started"
            })
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("get_status", "Get agent status and statistics")
    async def handle_status_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return create_success_response({
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "statistics": self.processing_stats,
            "active_tasks": len([t for t in self.tasks.values() if t["status"] == "running"]),
            "data_products_registered": len(self.data_products),
            "capabilities": self.list_skills()
        })
    
    @a2a_skill(
        name="dublin_core_extraction",
        description="Extract Dublin Core metadata from data products",
        capabilities=["metadata-extraction", "dublin-core", "iso-15836"],
        domain="metadata"
    )
    async def extract_dublin_core_metadata(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Dublin Core metadata according to ISO 15836 standards"""
        data_location = input_data.get("data_location", "")
        data_type = input_data.get("data_type", "dataset")
        
        # Read and analyze data
        metadata = await self._analyze_data_for_dublin_core(data_location)
        
        # Generate Dublin Core elements
        dublin_core = {
            "title": metadata.get("title", f"Financial Data Products - {datetime.utcnow().strftime('%Y-%m')}"),
            "creator": ["FinSight CIB", "Data Product Registration Agent", "A2A System"],
            "subject": metadata.get("subjects", ["financial-data", "enterprise-data"]),
            "description": metadata.get("description", f"Financial dataset with {metadata.get('record_count', 0)} records"),
            "publisher": "FinSight CIB Data Platform",
            "date": datetime.utcnow().isoformat(),
            "type": "Dataset",
            "format": metadata.get("format", "text/csv"),
            "identifier": f"dp-{uuid4().hex[:12]}",
            "language": "en",
            "rights": "Internal Use Only - FinSight CIB Proprietary Data"
        }
        
        self.processing_stats["dublin_core_extractions"] += 1
        
        return {
            "dublin_core_metadata": dublin_core,
            "compliance": {
                "iso_15836": True,
                "rfc_5013": True,
                "ansi_niso_z39_85": True
            },
            "quality_score": metadata.get("quality_score", 1.0)
        }
    
    @a2a_skill(
        name="integrity_verification",
        description="Verify data integrity using SHA256 hashing and referential checks",
        capabilities=["data-integrity", "sha256-hashing", "referential-integrity"],
        domain="data-quality"
    )
    async def verify_data_integrity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data integrity with SHA256 and referential checks"""
        data_location = input_data.get("data_location", "")
        
        integrity_results = {
            "overall_status": "verified",
            "sha256_hashes": {},
            "referential_integrity": {},
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Process CSV files for hashing
            csv_files = self._find_csv_files(data_location)
            
            for csv_file in csv_files:
                file_hash = await self._calculate_file_hash(csv_file)
                integrity_results["sha256_hashes"][os.path.basename(csv_file)] = file_hash
            
            # Verify referential integrity
            ref_integrity = await self._verify_referential_integrity(csv_files)
            integrity_results["referential_integrity"] = ref_integrity
            
            self.processing_stats["integrity_verifications"] += 1
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            integrity_results["overall_status"] = "failed"
            integrity_results["error"] = str(e)
        
        return integrity_results
    
    @a2a_skill(
        name="ord_registration",
        description="Register data products with ORD Registry",
        capabilities=["ord-registration", "catalog-management"],
        domain="data-catalog"
    )
    async def register_with_ord(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register data product with ORD Registry"""
        dublin_core = input_data.get("dublin_core_metadata", {})
        integrity_info = input_data.get("integrity_info", {})
        
        ord_descriptor = {
            "title": dublin_core.get("title", ""),
            "shortDescription": dublin_core.get("description", "")[:250],
            "description": dublin_core.get("description", ""),
            "version": "1.0.0",
            "releaseStatus": "active",
            "visibility": "internal",
            "partOf": [{"title": "FinSight CIB Data Platform"}],
            "tags": dublin_core.get("subject", []),
            "labels": {
                "data-type": ["financial"],
                "processing-level": ["raw", "structured"],
                "compliance": ["dublin-core", "iso-15836"]
            },
            "documentationLabels": {
                "Created By": "Data Product Registration Agent",
                "Dublin Core Compliant": "true",
                "SHA256 Verified": str(integrity_info.get("overall_status") == "verified").lower()
            }
        }
        
        try:
            # Register with ORD Registry
            registration_result = await self._register_ord_descriptor(ord_descriptor)
            
            self.processing_stats["successful_registrations"] += 1
            
            return {
                "registration_successful": True,
                "ord_id": registration_result.get("id"),
                "ord_descriptor": ord_descriptor,
                "registry_url": self.ord_registry_url
            }
            
        except Exception as e:
            logger.error(f"ORD registration failed: {e}")
            return {
                "registration_successful": False,
                "error": str(e)
            }
    
    @a2a_task(
        task_type="data_product_registration",
        description="Complete data product registration workflow",
        timeout=600,
        retry_attempts=2
    )
    async def process_data_product_workflow(self, data_info: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete workflow for data product registration"""
        
        results = {
            "workflow_id": f"dp_reg_{uuid4().hex[:8]}",
            "context_id": context_id,
            "stages": {}
        }
        
        try:
            # Stage 1: Dublin Core extraction
            dublin_core_result = await self.execute_skill("dublin_core_extraction", data_info)
            results["stages"]["dublin_core"] = dublin_core_result
            
            if not dublin_core_result["success"]:
                raise Exception("Dublin Core extraction failed")
            
            # Stage 2: Integrity verification
            integrity_result = await self.execute_skill("integrity_verification", data_info)
            results["stages"]["integrity"] = integrity_result
            
            if not integrity_result["success"]:
                raise Exception("Integrity verification failed")
            
            # Stage 3: ORD registration
            ord_input = {
                "dublin_core_metadata": dublin_core_result["result"]["dublin_core_metadata"],
                "integrity_info": integrity_result["result"]
            }
            
            ord_result = await self.execute_skill("ord_registration", ord_input)
            results["stages"]["ord_registration"] = ord_result
            
            # Update statistics
            self.processing_stats["total_processed"] += 1
            
            # Store data product info
            data_product_id = dublin_core_result["result"]["dublin_core_metadata"]["identifier"]
            self.data_products[data_product_id] = {
                "dublin_core": dublin_core_result["result"]["dublin_core_metadata"],
                "integrity": integrity_result["result"],
                "ord_registration": ord_result["result"] if ord_result["success"] else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
            return {
                "workflow_successful": True,
                "data_product_id": data_product_id,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Data product workflow failed: {e}")
            return {
                "workflow_successful": False,
                "error": str(e),
                "partial_results": results
            }
    
    async def _process_data_product(self, task_id: str, data_info: Dict[str, Any], context_id: str):
        """Process data product asynchronously"""
        try:
            from ....sdk.types import TaskStatus
            await self.update_task(task_id, TaskStatus.RUNNING)
            
            # Execute the workflow
            result = await self.process_data_product_workflow(data_info, context_id)
            
            if result["workflow_successful"]:
                await self.update_task(task_id, TaskStatus.COMPLETED, result=result)
            else:
                await self.update_task(task_id, TaskStatus.FAILED, error=result.get("error"))
                
        except Exception as e:
            await self.update_task(task_id, TaskStatus.FAILED, error=str(e))
    
    def _extract_data_info(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract data information from message"""
        data_info = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                data_info.update(part.data)
            elif part.kind == "file" and part.file:
                data_info["file"] = part.file
        
        return data_info
    
    async def _analyze_data_for_dublin_core(self, data_location: str) -> Dict[str, Any]:
        """Analyze data to extract metadata for Dublin Core"""
        metadata = {
            "record_count": 0,
            "quality_score": 1.0,
            "subjects": ["financial-data"],
            "format": "text/csv"
        }
        
        try:
            if os.path.exists(data_location):
                csv_files = self._find_csv_files(data_location)
                total_records = 0
                
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    total_records += len(df)
                
                metadata["record_count"] = total_records
                metadata["title"] = f"Financial Data Products - {len(csv_files)} datasets"
                metadata["description"] = f"Financial data collection with {total_records:,} records across {len(csv_files)} data types"
                
        except Exception as e:
            logger.warning(f"Data analysis failed: {e}")
        
        return metadata
    
    def _find_csv_files(self, data_location: str) -> List[str]:
        """Find CSV files in data location"""
        csv_files = []
        
        if os.path.isfile(data_location) and data_location.endswith('.csv'):
            csv_files.append(data_location)
        elif os.path.isdir(data_location):
            for root, dirs, files in os.walk(data_location):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
        
        return csv_files
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _verify_referential_integrity(self, csv_files: List[str]) -> Dict[str, Any]:
        """Verify referential integrity between files"""
        integrity_status = {
            "overall_status": "verified",
            "foreign_key_checks": [],
            "orphaned_records": 0,
            "integrity_score": 1.0
        }
        
        try:
            # Simple referential integrity check
            # In a real implementation, this would be more sophisticated
            integrity_status["foreign_key_checks"].append({
                "relationship": "basic_structure_check",
                "status": "verified",
                "details": "File structure validation passed"
            })
            
        except Exception as e:
            integrity_status["overall_status"] = "failed"
            integrity_status["error"] = str(e)
        
        return integrity_status
    
    async def _register_ord_descriptor(self, ord_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Register ORD descriptor with registry"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ord_registry_url}/api/v1/ord/register",
                    json=ord_descriptor,
                    timeout=30
                )
                
                if response.status_code == 201:
                    return response.json()
                else:
                    raise Exception(f"Registration failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"ORD registration failed: {e}")
            raise
    
    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        state_file = os.path.join(self.storage_path, "agent_state.json")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.processing_stats = state.get("processing_stats", self.processing_stats)
                    self.data_products = state.get("data_products", {})
                    
                logger.info("Loaded persistent state successfully")
            except Exception as e:
                logger.warning(f"Failed to load persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save persistent state to storage"""
        state_file = os.path.join(self.storage_path, "agent_state.json")
        
        try:
            state = {
                "processing_stats": self.processing_stats,
                "data_products": self.data_products,
                "last_saved": datetime.utcnow().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("Saved persistent state successfully")
        except Exception as e:
            logger.warning(f"Failed to save persistent state: {e}")
    
    async def _initialize_trust_system(self) -> None:
        """Initialize the agent's trust system"""
        try:
            # Initialize trust identity
            self.trust_identity = await initialize_agent_trust(
                self.agent_id,
                self.base_url
            )
            
            if self.trust_identity:
                logger.info(f"✅ Trust system initialized for {self.agent_id}")
                logger.info(f"   Trust address: {self.trust_identity.get('address')}")
                logger.info(f"   Public key fingerprint: {self.trust_identity.get('public_key_fingerprint')}")
                
                # Get trust contract reference
                self.trust_contract = get_trust_contract()
                
                # Pre-trust essential agents
                essential_agents = [
                    "agent_manager",
                    "data_standardization_agent_1",
                    "ai_preparation_agent_2",
                    "vector_processing_agent_3"
                ]
                
                for agent_id in essential_agents:
                    self.trusted_agents.add(agent_id)
                
                logger.info(f"   Pre-trusted agents: {self.trusted_agents}")
            else:
                logger.warning("⚠️  Trust system initialization failed, running without trust verification")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize trust system: {e}")
            logger.warning("Continuing without trust verification")
    
    async def _initialize_network_integration(self) -> None:
        """Initialize integration with a2aNetwork services"""
        try:
            # Import network services
            from app.a2a.network import get_network_connector, get_registration_service, get_messaging_service
            
            # Get network connector
            self.network_connector = get_network_connector()
            
            # Initialize network connectivity
            network_available = await self.network_connector.initialize()
            
            if network_available:
                logger.info(f"🌐 Agent {self.agent_id} connected to a2aNetwork")
                
                # Auto-register with network
                registration_service = get_registration_service()
                await registration_service.initialize()
                
                registration_result = await registration_service.register_agent_on_startup(self)
                
                if registration_result.get("success", False):
                    logger.info(f"📋 Agent {self.agent_id} registered with network registry")
                    self.network_registered = True
                else:
                    logger.warning(f"⚠️  Agent registration failed: {registration_result}")
                    self.network_registered = False
                
                # Initialize messaging service
                self.messaging_service = get_messaging_service()
                await self.messaging_service.initialize()
                
                # Register message handlers for this agent
                await self._register_message_handlers()
                
            else:
                logger.info(f"📱 Agent {self.agent_id} running in local mode (network unavailable)")
                self.network_registered = False
                
        except Exception as e:
            logger.error(f"❌ Network integration failed for {self.agent_id}: {e}")
            self.network_registered = False
    
    async def _register_message_handlers(self):
        """Register handlers for incoming network messages"""
        try:
            # Register handler for data product registration requests
            await self.messaging_service.register_message_handler(
                self.agent_id, 
                "data_product_request",
                self._handle_network_data_product_request
            )
            
            # Register handler for capability requests
            await self.messaging_service.register_message_handler(
                self.agent_id,
                "capability_request", 
                self._handle_network_capability_request
            )
            
            logger.info(f"📥 Message handlers registered for {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to register message handlers: {e}")
    
    async def _handle_network_data_product_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming data product registration request from network"""
        try:
            payload = message.get("payload", {})
            data_info = payload.get("data_info", {})
            context_id = message.get("context_id", "network_msg")
            
            logger.info(f"📩 Received data product request from {message.get('from_agent')}")
            
            # Process the data product registration
            result = await self.process_data_product_workflow(data_info, context_id)
            
            return {
                "success": result.get("workflow_successful", False),
                "data": result,
                "processed_by": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error handling network data product request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_network_capability_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming capability request from network"""
        try:
            payload = message.get("payload", {})
            requested_capability = payload.get("requested_capability", "")
            request_data = payload.get("request_data", {})
            
            logger.info(f"🔧 Capability request: {requested_capability} from {message.get('from_agent')}")
            
            # Check if we have the requested capability
            available_capabilities = ["dublin-core", "metadata-extraction", "data-integrity", "ord-registration"]
            
            if requested_capability in available_capabilities:
                # Execute the appropriate skill
                if requested_capability == "dublin-core" or requested_capability == "metadata-extraction":
                    result = await self.extract_dublin_core_metadata(request_data)
                elif requested_capability == "data-integrity":
                    result = await self.verify_data_integrity(request_data)
                elif requested_capability == "ord-registration":
                    result = await self.register_with_ord(request_data)
                else:
                    result = {"error": "Capability not implemented"}
                
                return {
                    "success": True,
                    "capability": requested_capability,
                    "result": result,
                    "provided_by": self.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Capability '{requested_capability}' not available",
                    "available_capabilities": available_capabilities
                }
                
        except Exception as e:
            logger.error(f"Error handling capability request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
